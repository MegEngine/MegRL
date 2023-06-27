#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import multiprocessing as mp
import os
import time
from loguru import logger

import megengine.distributed as dist
from megengine.device import get_device_count

from basecore.utils import get_free_port, get_hostip, init_local_pg


__all__ = ["launch"]


def launch(
    func, args=(), kwargs={}, devices_per_machine=None,
    num_machines=1, machine_rank=0, mp_start_method=None,
):
    """
    megengine distributed launch function. Calling like `launch(func, (1,), {name=2})`
    is equal to `func(1, name=2)` with distributed wrapper.

    Args:
        func (Callable): name of function to launch.
        args (tuple): arguments of function.
        kwargs (dict): keyword arguments of function.
        devices_per_machine (int): number of device on every machine. Default to None, means
            device_count is used.
        num_machines (int): number of used machines. Default to 1.
        machine_rank (int): rank of current machine. Default to 0.
        mp_start_method (string): multiprocessing start method. 'fork', 'spwan', 'forkserver'
            is supported. Defaults to None, means method is set depend on your OS.
    """
    if devices_per_machine is None:
        devices_per_machine = get_device_count("gpu")
    world_size = num_machines * devices_per_machine

    if mp_start_method is not None:
        mp.set_start_method(method=mp_start_method)

    # launch processes
    if world_size > 1:
        if machine_rank == 0:
            # create server if is master
            ip_addr, port = get_hostip()["IP"], get_free_port()
            # do not write code like `dist.Server`, otherwise python will gc server.
            server = dist.Server(port=port)  # noqa

            if num_machines > 1:
                # NOTE write address and port info to local file. Only useful for brainpp
                with open("addr_and_port.txt", "w") as f:
                    f.write(f"{ip_addr}:{port}")
        else:
            # NOTE read local file to get addr and port. Only useful for brainpp
            content = _read_local_file("addr_and_port.txt")
            ip_addr, port = content.split(":")  # content example: 123.123.123.123:12123
            port = int(port)

        procs = []
        for local_rank in range(devices_per_machine):
            global_rank = machine_rank * devices_per_machine + local_rank
            p = mp.Process(
                target=_dist_worker,
                kwargs=dict(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    ip_addr=ip_addr,
                    port=port,
                    rank=global_rank,
                    world_size=world_size,
                    num_machines=num_machines,
                    machine_rank=machine_rank,
                ),
            )
            p.start()
            procs.append(p)

        # join processes
        for p in procs:
            p.join()
    else:
        func(*args, **kwargs)


def _dist_worker(
    func,
    args,
    kwargs,
    ip_addr,
    port,
    rank=0,
    world_size=1,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
):
    try:
        devices_per_machine = world_size // num_machines
        dist.init_process_group(
            master_ip=ip_addr,
            port=port,
            world_size=world_size,
            rank=rank,
            device=rank % devices_per_machine,
            backend=backend,
        )
        logger.info(f"init process group rank {rank}/{world_size}")
        dist.group_barrier()
    except Exception:
        logger.info("init process group failed...")
        raise

    local_ranks = list(
        range(machine_rank * devices_per_machine, (machine_rank + 1) * devices_per_machine)
    )
    init_local_pg(local_ranks)

    func(*args, **kwargs)


def _read_local_file(filename, max_try=3):
    for i in range(max_try):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return f.read()
        else:
            # exp sleep time
            time.sleep(2 ** (i + 1))
    raise ValueError("Could read from {}".format(filename))
