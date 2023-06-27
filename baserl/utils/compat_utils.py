#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine as mge

_MGE_VER = [int(x) for x in mge.__version__.split(".")[:2]]


def get_device_count(device_type: str) -> int:
    """
    Make sure that `get_device_count` function is called correctly in BaseCore.
    """
    # breaking change since MGE 1.5.0
    if _MGE_VER >= [1, 5]:
        return mge.device.get_device_count(device_type)
    else:
        return mge.distributed.helper.get_device_count_by_fork(device_type)
