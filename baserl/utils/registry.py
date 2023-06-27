#!/usr/bin/python3
# -*- coding:utf-8 -*-

import inspect

from basecore.utils import Registry


def is_type(obj, parent_type):
    return inspect.isclass(obj) and issubclass(obj, parent_type)


class registers:
    """All registried module could be found here."""
    # data related registry
    datasets = Registry("datasets")
    datasets_info = Registry("datasets info")
    transforms = Registry("transforms")


def register_mge_transform():
    import megengine.data.transform as T
    transforms = registers.transforms
    for name, obj in vars(T).items():
        if is_type(obj, T.Transform):
            transforms.register(obj, name="MGE_" + name)


def register_mge_dataset():
    import megengine.data.dataset as D
    datasets = registers.datasets
    for name, obj in vars(D).items():
        if is_type(obj, D.VisionDataset):
            datasets.register(obj, name=name)


def all_register():
    # try logic is used to avoid AssertionError of register twice
    try:
        register_mge_transform()
        register_mge_dataset()
    except AssertionError:
        pass
