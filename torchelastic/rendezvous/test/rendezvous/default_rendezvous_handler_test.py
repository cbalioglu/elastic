# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from datetime import timedelta
from typing import Optional
from unittest import TestCase

from torchelastic.rendezvous import RendezvousParameters
from torchelastic.rendezvous.default_rendezvous_handler import (
    DefaultRendezvousHandler,
    RendezvousBackend,
    RendezvousTimeout,
    _WorkerId,
    _WorkerIdGenerator,
    create_handler,
)


class WorkerIdTest(TestCase):
    def test_repr(self) -> None:
        worker_id = _WorkerId("hostname", 3, 5)

        self.assertEqual(repr(worker_id), "hostname_3_5")

    def test_hash(self) -> None:
        worker_id1 = _WorkerId("hostname", 2, 4)
        worker_id2 = _WorkerId("hostname", 3, 5)

        worker_ids = {worker_id1, worker_id2}

        self.assertIn(worker_id1, worker_ids)
        self.assertIn(worker_id2, worker_ids)


class WorkerIdGeneratorTest(TestCase):
    def test_generate(self) -> None:
        worker_id_generator = _WorkerIdGenerator()

        hostname = socket.getfqdn()

        pid = os.getpid()

        for local_id in range(4):
            with self.subTest(hostname=hostname, pid=pid, local_id=local_id):
                worker_id = worker_id_generator.generate()

                self.assertEqual(repr(worker_id), f"{hostname}_{pid}_{local_id}")


class RendezvousTimeoutTest(TestCase):
    # fmt: off

    def test_init_initializes_timeout(self) -> None:
        timeout = RendezvousTimeout(
            timedelta(seconds=50),
            timedelta(seconds=60),
            timedelta(seconds=70),
            timedelta(seconds=80),
        )

        self.assertEqual(timeout.join,      timedelta(seconds=50))
        self.assertEqual(timeout.last_call, timedelta(seconds=60))
        self.assertEqual(timeout.close,     timedelta(seconds=70))
        self.assertEqual(timeout.store,     timedelta(seconds=80))

    def test_init_initializes_timeout_if_no_timeout_is_specified(self) -> None:
        timeout = RendezvousTimeout()

        self.assertEqual(timeout.join,      timedelta(seconds=600))
        self.assertEqual(timeout.last_call, timedelta(seconds=30))
        self.assertEqual(timeout.close,     timedelta(seconds=30))
        self.assertEqual(timeout.store,     timedelta(seconds=300))

    # fmt: on


class DummyRendezvousBackend(RendezvousBackend):
    def get_state(self):
        return None

    def set_state(self, state, token):
        return None

    @property
    def name(self) -> str:
        return "dummy_backend"


class DefaultRendezvousHandlerTest(TestCase):
    def setUp(self) -> None:
        self._run_id = "dummy_run_id"
        self._backend = DummyRendezvousBackend()
        self._min_num_participants = 3
        self._max_num_participants = 6
        self._store_port: Optional[int] = 1234
        self._timeout: Optional[RendezvousTimeout] = RendezvousTimeout()

    def _create_handler(self) -> DefaultRendezvousHandler:
        kwargs = {}

        if self._store_port is not None:
            kwargs["store_port"] = self._store_port

        return DefaultRendezvousHandler(
            run_id=self._run_id,
            backend=self._backend,
            min_num_participants=self._min_num_participants,
            max_num_participants=self._max_num_participants,
            timeout=self._timeout,
            **kwargs,
        )

    def test_init_initializes_handler(self) -> None:
        handler = self._create_handler()

        if self._store_port is not None:
            self._expected_store_port = self._store_port

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._run_id)
        self.assertEqual(handler.min_num_participants, self._min_num_participants)
        self.assertEqual(handler.max_num_participants, self._max_num_participants)
        self.assertEqual(handler.store_port, self._expected_store_port)

        if self._timeout is None:
            self.assertIsNotNone(handler.timeout)
        else:
            self.assertIs(handler.timeout, self._timeout)

    def test_init_initializes_handler_if_store_port_is_not_specified(self) -> None:
        self._store_port = None

        self._expected_store_port = 29500

        self.test_init_initializes_handler()

    def test_init_initializes_handler_if_timeout_is_not_specified(self) -> None:
        self._timeout = None

        self.test_init_initializes_handler()

    def test_init_initializes_handler_if_min_and_max_num_participants_are_equal(self) -> None:
        self._min_num_participants = 3
        self._max_num_participants = 3

        self.test_init_initializes_handler()

    def test_init_raises_error_if_min_num_participants_is_non_positive(self) -> None:
        for num in [0, -10]:
            with self.subTest(min_num_participants=num):
                self._min_num_participants = num

                with self.assertRaisesRegex(
                    ValueError, r"^The minimum number of participants must be greater than zero.$"
                ):
                    self._create_handler()

    def test_init_raises_error_if_max_num_participants_is_less_than_min(self) -> None:
        self._min_num_participants = 3
        self._max_num_participants = 2

        with self.assertRaisesRegex(
            ValueError,
            r"^The maximum number of participants must be greater than or equal to the minimum number of participants.$",
        ):
            self._create_handler()

    def test_init_raises_error_if_store_port_is_greater_than_or_equal_to_65536(self) -> None:
        for store_port in [65536, 70000]:
            with self.subTest(store_port=store_port):
                self._store_port = store_port

                with self.assertRaisesRegex(
                    ValueError, r"^The port number of the store must be less than 65536.$"
                ):
                    self._create_handler()


class CreateHandlerTest(TestCase):
    def setUp(self) -> None:
        self._backend = DummyRendezvousBackend()

        self._params = RendezvousParameters(
            backend=self._backend.name,
            endpoint="dummy_end_point",
            run_id="dummy_run_id",
            min_nodes=3,
            max_nodes=6,
            store_port="1234",
            join_timeout="50",
            last_call_timeout="60",
            close_timeout="70",
            store_timeout="80",
        )

        self._expected_store_port = 1234
        self._expected_close_timeout = 70

    def test_create_handler_returns_handler(self) -> None:
        handler = create_handler(self._backend, self._params)

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._params.run_id)
        self.assertEqual(handler.min_num_participants, self._params.min_nodes)
        self.assertEqual(handler.max_num_participants, self._params.max_nodes)
        self.assertEqual(handler.store_port, self._expected_store_port)
        self.assertEqual(handler.timeout.join, timedelta(seconds=50))
        self.assertEqual(handler.timeout.last_call, timedelta(seconds=60))
        self.assertEqual(handler.timeout.close, timedelta(seconds=self._expected_close_timeout))
        self.assertEqual(handler.timeout.store, timedelta(seconds=80))

    def test_create_handler_returns_handler_if_store_port_is_not_specified(self) -> None:
        del self._params.config["store_port"]

        self._expected_store_port = 29500

        self.test_create_handler_returns_handler()

    def test_create_handler_returns_handler_if_timeout_is_not_specified(self) -> None:
        del self._params.config["close_timeout"]

        self._expected_close_timeout = 30

        self.test_create_handler_returns_handler()

    def test_create_handler_raises_error_if_store_port_is_invalid(self) -> None:
        self._params.config["store_port"] = "dummy"

        with self.assertRaisesRegex(
            ValueError, r"^The store port number must be an integer between 0 and 65536.$"
        ):
            create_handler(self._backend, self._params)
