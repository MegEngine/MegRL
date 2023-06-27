import megengine as mge
import megengine.distributed as dist
import megengine.optimizer as optim

from basecore.engine import BaseTrainer

from baserl.layers import ModelEMA, calculate_momentum
from baserl.utils import MeterBuffer


class OffPolicyTrainer(BaseTrainer):
    """
    Attributes:
        progress: training process. Contains basic informat such as current iter, max iter.
        model: trained model.
        solver: solver that contains optimizer, grad_manager and so on.
        dataloader: data provider.
        meter: meters to log, such as train_time, losses.
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        Args:
            cfg (Config): config which describes training process.
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        # self.dataloader_iter = iter(self.dataloader)
        self.config_to_attr()
        self.meter = MeterBuffer(window_size=self.meter_window_size)
        self.reward_meter = MeterBuffer(window_size=self.reward_meter_window_size)

        if isinstance(self.solver, dict):
            def modify_grad():
                if self.enable_grad_clip:
                    params = self.model.parameters()
                    clip_type = self.grad_clip_type
                    clip_func = optim.clip_grad_value if clip_type == "value" else optim.clip_grad_norm
                    clip_func(params, **self.grad_clip_args)
            for k, sol in self.solver.items():
                model_name = k.replace('solver_', '')
                sol.grad_clip_fn = modify_grad

        else:
            def modify_grad():
                if self.enable_grad_clip:
                    params = self.model.parameters()
                    clip_type = self.grad_clip_type
                    clip_func = optim.clip_grad_value if clip_type == "value" else optim.clip_grad_norm
                    clip_func(params, **self.grad_clip_args)

            self.solver.grad_clip_fn = modify_grad
            if self.enable_amp:
                assert self.solver.grad_scaler is not None, "enable AMP but grad_scaler is None"

        self.last_len = 0
        self.last_rew = 0
        self.env_step = 0

    def config_to_attr(self):
        self.meter_window_size = self.cfg.GLOBAL.LOG_INTERVAL
        self.max_epoch = self.cfg.SOLVER.MAX_EPOCH
        self.num_env_step_per_epoch = self.cfg.SOLVER.NUM_ENV_STEP_PER_EPOCH
        self.num_train_step_per_epoch = int(self.num_env_step_per_epoch * self.cfg.MODEL.UPDATE_PER_STEP)
        self.enable_amp = self.cfg.TRAINER.AMP.ENABLE

        grad_clip_cfg = self.cfg.TRAINER.GRAD_CLIP
        self.enable_grad_clip = grad_clip_cfg.ENABLE
        if self.enable_grad_clip:
            self.grad_clip_type = grad_clip_cfg.TYPE
            assert self.grad_clip_type in ("value", "norm")
            self.grad_clip_args = grad_clip_cfg.ARGS

        ema_config = self.cfg.TRAINER.get("EMA", None)
        self.enable_ema = False if ema_config is None else ema_config.ENABLE
        if self.enable_ema:
            momentum = ema_config.MOMENTUM
            if momentum is None:
                total_iter = self.max_epoch * self.num_train_step_per_epoch # / self.model_batch_size
                update_period = ema_config.UPDATE_PERIOD
                momentum = calculate_momentum(ema_config.ALPHA, total_iter, update_period)
            self.ema = ModelEMA(self.model, momentum, burnin_iter=ema_config.BURNIN_ITER)
        
        self.reward_meter_window_size = self.max_epoch * 2 
        if self.cfg.MODEL.EPS_SCHEDULE_LENGTH is not None:
            self.eps_iter_thresh = self.cfg.MODEL.EPS_SCHEDULE_LENGTH // dist.get_world_size()
        else:
            self.eps_iter_thresh = None

    def train(self, start_training_info=(1, 1), max_training_info=None):
        if max_training_info is None:
            max_iter = int(
                self.num_train_step_per_epoch / dist.get_world_size()
            )
            max_training_info = (self.max_epoch, max_iter)
        super().train(start_training_info, max_training_info)

    @classmethod
    def data_to_input(cls, data):
        return {}

    def train_one_iter(self):
        """basic logic of training one iteration."""
        self.policy.model.train()
        # parameters
        reward_metric = self.cfg.MODEL.REWARD_METRIC
        update_per_step = self.cfg.MODEL.UPDATE_PER_STEP
        step_per_collect = self.cfg.MODEL.STEP_PER_COLLECT
        batch_size = self.cfg.TRAINER.BATCH_SIZE
        EPS_end = self.cfg.MODEL.EPS_END
        EPS_begin = self.cfg.MODEL.EPS_BEGIN

        self.model.batch_size = batch_size

        # initialized numbers
        
        gradient_step = 0
        result = self.dataloader.collect(n_step=step_per_collect)

        if result["n/ep"] > 0 and reward_metric:
            rew = reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.last_rew = result['rew'] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result['len'] if result["n/ep"] > 0 else self.last_len
        
        performance_meters = {
            'len_loss' : int(self.last_len),
            'reward_loss' : float(self.last_rew),
            'env_step_loss' : int(self.env_step),
            'update_times' : round(update_per_step * result["n/st"])
        }

        self.meter.update(**performance_meters)
        if hasattr(self.policy, 'set_eps'):
            eval("self.set_policy_eps_"+self.cfg.MODEL.EPS_SCHEDULE)(
                self.env_step, EPS_begin, EPS_end)

        for _ in range(round(update_per_step * result["n/st"])):
            gradient_step += 1

            self.mini_batch = {}

            if self.dataloader.buffer is None:
                continue

            batch, indices = self.dataloader.buffer.sample(batch_size)
            self.policy.updating = True
            batch = self.policy.process_fn(batch, self.dataloader.buffer, indices)

            obs = batch["obs"]
            obs_next = obs.obs if hasattr(obs, "obs") else obs
            net_inputs = {
                'obs' : mge.tensor(obs_next),
                'state' : None,
                'batch' : batch
            }

            if hasattr(self.policy, '_target'):
                if self.policy._target and self.policy._iter % self.policy._freq == 0:
                    self.policy.sync_weight()

            loss_dict = self.model_step(net_inputs)

            if hasattr(self.policy, '_iter'):
                self.policy._iter += 1

            self.policy.post_process_fn(batch, self.dataloader.buffer, indices)
            self.policy.updating = False


            mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution

            loss_meters = {name: float(loss) for name, loss in loss_dict.items()}

            time_meters = {}
            self.meter.update(**loss_meters, **time_meters)

    def model_step(self, model_inputs, model_name=None):
        """
        :meth:`model_step` should be called by :meth:`train_one_iter`, it defines
        basic logic of updating model's parameters.

        Args:
            model_inputs: input of models.
        """
        if model_name is not None and isinstance(self.solver, dict):
            sol = self.solver["solver_{}".format(model_name)]
            model = getattr(self.model, model_name)
            model_outputs = sol.minimize(model, model_inputs)
        else:
            model_outputs = self.solver.minimize(self.model, model_inputs)
        
        if self.enable_ema:
            self.ema.step()
        return model_outputs
    
    def before_train(self):
        for h in self._hooks:
            h.before_train()
        if self.cfg.MODEL.COLLECT_BEFORE_TRAIN and self.cfg.MODEL.COLLECT_BEFORE_TRAIN > 0:
            result = self.dataloader.collect(n_step=self.cfg.MODEL.COLLECT_BEFORE_TRAIN, 
                                             random=self.cfg.MODEL.COLLECT_BEFORE_TRAIN_RANDOM)
        
        return 

    def set_policy_eps_linear_decay(self, cur_step, EPS_begin, EPS_end):
        if cur_step <= self.eps_iter_thresh:
            eps = EPS_begin - cur_step / self.eps_iter_thresh * \
                (EPS_begin - EPS_end)
        else:
            eps = EPS_end
        self.policy.set_eps(eps)
    
    def set_policy_eps_fixed(self, cur_step, EPS_begin, EPS_end):
        self.policy.set_eps(EPS_end)
