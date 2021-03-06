# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from typing import ClassVar
from unittest import TestCase

from torch.distributed import TCPStore

from torchelastic.rendezvous import RendezvousConnectionError, RendezvousParameters
from torchelastic.rendezvous.c10d_rendezvous_backend import C10dRendezvousBackend, create_backend

from .rendezvous_backend_test import RendezvousBackendTestBase


class C10dRendezvousBackendTest(TestCase, RendezvousBackendTestBase):
    _store: ClassVar[TCPStore]

    @classmethod
    def setUpClass(cls) -> None:
        cls._store = TCPStore("127.0.0.1", 0, is_master=True)  # type: ignore[call-arg]

    def setUp(self) -> None:
        # Make sure we have a clean slate.
        self._store.delete_key("torch.elastic.rendezvous.dummy_run_id")

        self._backend = C10dRendezvousBackend(self._store, "dummy_run_id")

    def _corrupt_state(self) -> None:
        self._store.set("torch.elastic.rendezvous.dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    def setUp(self) -> None:
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint="127.0.0.1:29400",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            is_host="true",
            store_type="tCp",
            read_timeout="10",
        )

        self._expected_endpoint_host = "127.0.0.1"
        self._expected_endpoint_port = 29400
        self._expected_store_type = TCPStore
        self._expected_read_timeout = timedelta(seconds=10)

    def test_create_backend_returns_backend(self) -> None:
        backend = create_backend(self._params)

        self.assertIsInstance(backend.store, self._expected_store_type)

        self.assertEqual(backend.name, "c10d-experimental")
        self.assertEqual(backend.key, "torch.elastic.rendezvous." + self._params.run_id)

        store = backend.store

        self.assertEqual(store.host, self._expected_endpoint_host)  # type: ignore[attr-defined]
        self.assertEqual(store.port, self._expected_endpoint_port)  # type: ignore[attr-defined]
        self.assertEqual(store.timeout, self._expected_read_timeout)  # type: ignore[attr-defined]

    def test_create_backend_returns_backend_if_is_host_is_false(self) -> None:
        store = TCPStore(  # type: ignore[call-arg] # noqa: F841
            self._expected_endpoint_host, self._expected_endpoint_port, is_master=True
        )

        self._params.config["is_host"] = "false"

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified(self) -> None:
        del self._params.config["is_host"]

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified_and_store_already_exists(
        self,
    ) -> None:
        store = TCPStore(  # type: ignore[call-arg] # noqa: F841
            self._expected_endpoint_host, self._expected_endpoint_port, is_master=True
        )

        del self._params.config["is_host"]

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_port_is_not_specified(self) -> None:
        self._params.endpoint = self._expected_endpoint_host

        self._expected_endpoint_port = 29500

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_store_type_is_not_specified(self) -> None:
        del self._params.config["store_type"]

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(self) -> None:
        del self._params.config["read_timeout"]

        self._expected_read_timeout = timedelta(seconds=60)

        self.test_create_backend_returns_backend()

    def test_create_backend_raises_error_if_store_is_unreachable(self) -> None:
        self._params.config["is_host"] = "false"
        self._params.config["read_timeout"] = "2"

        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to the C10d store has failed. See inner exception for details.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_endpoint_is_invalid(self) -> None:
        for is_host in [True, False]:
            with self.subTest(is_host=is_host):
                self._params.config["is_host"] = str(is_host)

                self._params.endpoint = "dummy_endpoint"

                with self.assertRaisesRegex(
                    RendezvousConnectionError,
                    r"^The connection to the C10d store has failed. See inner exception for "
                    r"details.$",
                ):
                    create_backend(self._params)

    def test_create_backend_raises_error_if_store_type_is_invalid(self) -> None:
        self._params.config["store_type"] = "dummy_store_type"

        with self.assertRaisesRegex(
            ValueError, r"^The store type must be 'tcp'. Other store types are not supported yet.$"
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                self._params.config["read_timeout"] = read_timeout

                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    create_backend(self._params)
