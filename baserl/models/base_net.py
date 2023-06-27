from abc import abstractmethod
from typing import Dict

import megengine as mge
import megengine.module as M
from megengine import Tensor

from baserl.utils import MeterBuffer, load_matched_weights


class BaseNet(M.Module):
    """Basic class for any network.

    Attributes:
        cfg (dict): a dict contains all settings for the network
    """

    def __init__(self):
        super().__init__()
        # extra_meter is used for logging extra used meter, such as accuracy.
        # user could use self.extra_meter.update(dict) to logging more info in basedet.
        self.extra_meter = MeterBuffer()

    @classmethod
    def preprocess_image(cls, data):
        """
        preprocess image for network

        Args:
            data: output data from dataloader
        """
        return {
            "image": mge.tensor(data[0]),
            "gt_mask": mge.tensor(data[1]),
            "img_info": data[2],
        }

    @abstractmethod
    def network_forward(self, inputs):
        """
        pure network forward logic
        """
        pass

    def forward(self, inputs):
        # return self.network_forward(inputs) 
        # for ActorDisA2C, its forward will be called in PGPolicy.forward ... 
        if self.training:
            return self.get_losses(inputs)
        else:
            return self.inference(inputs)

    @abstractmethod
    def get_losses(self, inputs) -> Dict[str, Tensor]:
        """ create(if have not create before) and return a dict which includes
        the whole losses within network.

        .. note::
            1. It must contains the `total` which indicates the total_loss in
               the returned loss dictionaries.
            2. Returned loss type must be OrderedDict.

        Args:
            inputs (dict[str, Tensor])

        Returns:
            loss_dict (OrderedDict[str, Tensor]): the OrderedDict contains the losses
        """
        pass

    @abstractmethod
    def inference(self, outputs, inputs=None) -> Dict[str, Tensor]:
        """Run inference for network

        Args:
            inputs (dict[str, Tensor])
        """
        pass

    def load_weights(self, weight_path: str, strict: bool = False) -> M.Module:
        """set weights of the network with the weight_path

        Args:
            weight_path (str): a file path of the weights
        """
        return load_matched_weights(self, weight_path, strict)

    def dump_weights(self, dump_path):
        mge.save({"state_dict": self.state_dict()}, dump_path)
