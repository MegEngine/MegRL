from numbers import Number, Real
from re import L
from typing import List, Union

import numpy as np
import math

from PIL import Image
import cv2
import megengine as mge
import megengine.functional as F
from baserl.layers import safelog


def normal_rsample(logits):
    loc, scale = logits
    # print(loc.shape)
    eps = mge.random.normal(size=scale.shape)
    # print(eps[0])
    return loc + eps * scale

def normal_log_prob(value, logits):
    loc, scale = logits
    var = (scale ** 2)
    log_scale = math.log(scale) if isinstance(scale, Real) else safelog(scale)
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

def sum_rightmost(value, dim):
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)

def independent_log_prob(base_prob, reinterpreted_dims):
    return sum_rightmost(base_prob, reinterpreted_dims)

def categorial_log_prob(value, logits):
    log_logits = safelog(logits)
    log_prob = log_logits[
        np.arange(len(log_logits)), value
    ]
    return log_prob

def categorial_entropy(logits):
    p_log_p = safelog(logits) * logits
    entropy = -p_log_p.sum(-1)
    return entropy

def logits_to_probs(logits):
    return F.softmax(logits, axis=-1)

def save_video(frames, path, duration):
    images = []
    for frame in frames:
        images.append(Image.fromarray(frame))

    images[0].save(path,
                save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)


class RunningMeanStd(object):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0
    ) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
