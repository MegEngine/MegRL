#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from megfile import SmartPath, s3

from refile import NoriPath


def internal_register():
    pass


def internal_setup():
    # set megfile
    s3.endpoint_url = "http://oss.i.brainpp.cn"
    if "nori" not in SmartPath._registered_protocols:
        if not hasattr(NoriPath, "protocol"):
            NoriPath.protocol = NoriPath.get_protocol()
        SmartPath.register(NoriPath)

    internal_register()


internal_setup()
