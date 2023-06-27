import time

import megengine as mge
import megengine.distributed as dist
import megengine.optimizer as optim

from basecore.engine import BaseTrainer

from baserl.layers import ModelEMA, calculate_momentum
from baserl.utils import MeterBuffer


class OnPolicyTrainer(BaseTrainer):
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
        self.num_env_episode_per_epoch = self.cfg.SOLVER.NUM_ENV_EPISODE_PER_EPOCH
        if self.num_env_step_per_epoch is not None:
            assert self.num_env_episode_per_epoch is None
        else:
            assert self.num_env_episode_per_epoch is not None
        self.model_batch_size = self.cfg.MODEL.BATCHSIZE_PER_DEVICE
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
                assert self.num_env_episode_per_epoch is not None 
                total_iter = self.max_epoch * self.num_env_episode_per_epoch / self.model_batch_size
                update_period = ema_config.UPDATE_PERIOD
                momentum = calculate_momentum(ema_config.ALPHA, total_iter, update_period)
            self.ema = ModelEMA(self.model, momentum, burnin_iter=ema_config.BURNIN_ITER)

        self.step_per_collect = self.cfg.MODEL.STEP_PER_COLLECT
        self.episode_per_collect = self.cfg.MODEL.EPISODE_PER_COLLECT
        self.repeat_per_collect = self.cfg.MODEL.REPEAT_PER_COLLECT
        self.reward_metric = self.cfg.MODEL.REWARD_METRIC
        self.batch_size = self.cfg.TRAINER.BATCH_SIZE
        
        self.reward_meter_window_size = self.max_epoch * 2
        
        self.recompute_adv = self.cfg.MODEL.get("PPO_RECOMPUTE_ADV", False)

    def train(self, start_training_info=(1, 1), max_training_info=None):
        if max_training_info is None:
            if self.num_env_episode_per_epoch is not None:
                max_iter = int(
                    self.num_env_episode_per_epoch / dist.get_world_size() # / self.model_batch_size
                )
            else:
                max_iter = int(
                    self.num_env_step_per_epoch / dist.get_world_size() # / self.model_batch_size
                )
            max_training_info = (self.max_epoch, max_iter)
        super().train(start_training_info, max_training_info)

    @classmethod
    def data_to_input(cls, data):
        return {}

    def train_in_epoch(self):
        while not self.progress.reach_epoch_end():
            self.before_epoch()
            while not self.progress.reach_iter_end():  # iter training process
                self.before_iter()
                n_step_this_iter = self.train_one_iter()
                self.after_iter()
                if self.num_env_episode_per_epoch is not None:
                    self.progress.next_iter()
                else:
                    self.progress.iter += n_step_this_iter
                    self.prev_iter_len = n_step_this_iter # For ETA calculation in hooks
            self.after_epoch()
            self.progress.next_epoch()
    
    def train_one_iter(self):
        """basic logic of training one iteration."""

        self.policy.train()
        # parameters
        step_per_collect = self.step_per_collect
        episode_per_collect = self.episode_per_collect
        repeat_per_collect = self.repeat_per_collect
        reward_metric = self.reward_metric
        batch_size = self.batch_size

        self.model.batch_size = batch_size

        # initialized numbers
        

        data_tik = time.time()
        result = self.dataloader.collect(n_step=step_per_collect, 
                                         n_episode=episode_per_collect)
        if result["n/ep"] > 0 and reward_metric:
            rew = reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.last_rew = result['rew'] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result['len'] if result["n/ep"] > 0 else self.last_len

        performance_meters = {
            'len_loss' : int(self.last_len),
            'env_step_loss' : int(self.env_step),
            'ep_loss' : int(result["n/ep"])
        }

        self.meter.update(**performance_meters)
        self.mini_batch = {}
        model_inputs = self.data_to_input(self.mini_batch)
        data_tok = time.time()
        train_tik = time.time()

        if self.dataloader.buffer is None:
            return 

        batch, indices = self.dataloader.buffer.sample(0) # 0 means all
        self.policy.updating = True
        batch = self.policy.process_fn(batch,self.dataloader.buffer, indices)
        for _ in range(repeat_per_collect):
            if self.recompute_adv and _ > 0:
                batch = self.policy._compute_returns(batch, self.policy._buffer, self.policy._indices)
            for minibatch in batch.split(batch_size,
                merge_last=True):
                loss_dict = self.model_step(minibatch)
                loss_meters = {name: float(loss) for name, loss in loss_dict.items()}
                self.meter.update(**loss_meters)
        
        
        self.policy.post_process_fn(batch, self.dataloader.buffer, indices)
        self.policy.updating = False
        self.dataloader.reset_buffer(keep_statistics=True)

        mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
        train_tok = time.time()
        time_meters = {"train_time": train_tok - train_tik, "data_time": data_tok - data_tik}
        time_meters = {}
        self.meter.update(**time_meters)
        
        return result["n/st"]

    def model_step(self, model_inputs):
        """
        :meth:`model_step` should be called by :meth:`train_one_iter`, it defines
        basic logic of updating model's parameters.

        Args:
            model_inputs: input of models.
        """
        model_inputs['policy'] = self.policy
        model_outputs = self.solver.minimize(self.policy.get_loss, model_inputs)
                
        if self.enable_ema:
            self.ema.step()
        return model_outputs
