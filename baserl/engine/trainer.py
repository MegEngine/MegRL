import time

import megengine as mge
import megengine.distributed as dist
import megengine.optimizer as optim

from basecore.engine import BaseTrainer

from baserl.layers import ModelEMA, calculate_momentum
from baserl.utils import MeterBuffer


class Trainer(BaseTrainer):
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
        self.dataloader_iter = iter(self.dataloader)
        self.config_to_attr()
        self.meter = MeterBuffer(window_size=self.meter_window_size)

        def modify_grad():
            if self.enable_grad_clip:
                params = self.model.parameters()
                clip_type = self.grad_clip_type
                clip_func = optim.clip_grad_value if clip_type == "value" else optim.clip_grad_norm
                clip_func(params, **self.grad_clip_args)

        self.solver.grad_clip_fn = modify_grad
        if self.enable_amp:
            assert self.solver.grad_scaler is not None, "enable AMP but grad_scaler is None"

    def config_to_attr(self):
        self.meter_window_size = self.cfg.GLOBAL.LOG_INTERVAL
        self.max_epoch = self.cfg.SOLVER.MAX_EPOCH
        num_image_per_epoch = self.cfg.SOLVER.NUM_IMAGE_PER_EPOCH
        if num_image_per_epoch is None:
            # dataloader might be wrapped by InfiniteSampler
            num_image_per_epoch = len(self.dataloader.dataset)
        self.num_image_per_epoch = num_image_per_epoch
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
                total_iter = self.max_epoch * self.num_image_per_epoch / self.model_batch_size
                update_period = ema_config.UPDATE_PERIOD
                momentum = calculate_momentum(ema_config.ALPHA, total_iter, update_period)
            self.ema = ModelEMA(self.model, momentum, burnin_iter=ema_config.BURNIN_ITER)

    def train(self, start_training_info=(1, 1), max_training_info=None):
        if max_training_info is None:
            max_iter = int(
                self.num_image_per_epoch / dist.get_world_size() / self.model_batch_size
            )
            max_training_info = (self.max_epoch, max_iter)
        super().train(start_training_info, max_training_info)

    @classmethod
    def data_to_input(cls, data):
        return {
            "image": mge.tensor(data["data"]),
            "gt_boxes": mge.tensor(data["gt_boxes"]),
            "img_info": mge.tensor(data["im_info"]),
        }

    def train_one_iter(self):
        """basic logic of training one iteration."""
        data_tik = time.time()
        self.mini_batch = next(self.dataloader_iter)
        model_inputs = self.data_to_input(self.mini_batch)
        data_tok = time.time()

        train_tik = time.time()
        loss_dict = self.model_step(model_inputs)
        mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
        train_tok = time.time()

        loss_meters = {name: float(loss) for name, loss in loss_dict.items()}
        time_meters = {"train_time": train_tok - train_tik, "data_time": data_tok - data_tik}
        self.meter.update(**loss_meters, **time_meters)

    def model_step(self, model_inputs):
        """
        :meth:`model_step` should be called by :meth:`train_one_iter`, it defines
        basic logic of updating model's parameters.

        Args:
            model_inputs: input of models.
        """
        model_outputs = self.solver.minimize(self.model, model_inputs)
        if self.enable_ema:
            self.ema.step()
        return model_outputs
