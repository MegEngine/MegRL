#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np


def rand_in_range(low, high):
    return np.random.rand() * (high - low) + low
