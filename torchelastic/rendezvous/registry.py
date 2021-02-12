# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import RendezvousHandler, RendezvousHandlerFactory, RendezvousParameters
from .default_rendezvous_handler import create_handler


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_exp_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous_backend

    backend = etcd_rendezvous_backend.create_backend(params)

    return create_handler(backend, params)


def _create_exp_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import c10d_rendezvous_backend

    backend = c10d_rendezvous_backend.create_backend(params)

    return create_handler(backend, params)


_factory = RendezvousHandlerFactory()
_factory.register("etcd", _create_etcd_handler)
_factory.register("etcd-experimental", _create_exp_etcd_handler)
_factory.register("c10d-experimental", _create_exp_c10d_handler)


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    return _factory.create_handler(params)
