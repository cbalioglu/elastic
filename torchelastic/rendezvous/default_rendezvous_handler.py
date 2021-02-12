# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import random
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Final, Optional, Set, Tuple, Union, cast, final

from torch.distributed import Store, TCPStore

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousTimeoutError,
)
from .utils import _ClosableStore, _PeriodicTimer, _try_parse_port


@dataclass(eq=True, frozen=True)
class _WorkerId:
    """Identifies a worker in the rendezvous.

    Attributes:
        hostname:
            The hostname of the machine on which the worker runs.
        pid:
            The id of the process in which the worker runs.
        local_id:
            A process-wide unique id.
    """

    hostname: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.hostname}_{self.pid}_{self.local_id}"


class _WorkerIdGenerator:
    """Generates unique worker ids.

    A worker id is a combination of a hostname, a process id, and an
    auto-incremented integer that uniquely identifies a worker.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to `generate()`.
        self._local_id = 0

    def generate(self) -> _WorkerId:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _WorkerId(socket.getfqdn(), os.getpid(), local_id)


class _Rendezvous:
    """Represents the state of a rendezvous.

    A rendezvous is synced across the workers via a `RendezvousBackend`.

    Attributes:
        round:
            The current round of the rendezvous.
        store_host:
            The id of the worker that will host the store to be returned at the
            end of the `next_rendezvous()` call.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The date and time at which the current round of the rendezvous will
            be considered complete if it is still waiting for participants to
            join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of workers that are waiting to participate in the next round
            of the rendezvous.
        last_keep_alives:
            A dictionary containing each worker's last keep-alive time.
    """

    round: int
    store_host: Optional[_WorkerId]
    complete: bool
    deadline: Optional[datetime]
    closed: bool
    participants: Dict[_WorkerId, int]
    wait_list: Set[_WorkerId]
    last_keep_alives: Dict[_WorkerId, datetime]

    def __init__(self) -> None:
        self.round = 0
        self.store_host = None
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.last_keep_alives = {}


class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""

    UPDATE_KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    REMOVE_FROM_PARTICIPANTS = 4
    REMOVE_FROM_WAIT_LIST = 5
    MARK_RENDEZVOUS_COMPLETE = 6
    MARK_RENDEZVOUS_CLOSED = 7
    SYNC_RENDEZVOUS = 8
    ERROR_CLOSED = 9
    ERROR_TIMEOUT = 10
    FINISH = 11


Token = Any
"""Represents an opaque fencing token used by the rendezvous backend."""


class RendezvousBackend(ABC):
    """Represents a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of the backend."""

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """Gets the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            `None` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token]]:
        """Sets the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified `token` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified `token` does not match the fencing token stored in
            the backend, the state won't be updated; instead the existing state
            along with its fencing token will be returned to the caller.
          - If the specified `token` is `None`, the new state will be set only
            if there is no existing state in the backend. Either the new state
            or the existing state along with its fencing token will be returned
            to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to `get_state()` or `set_state()`.

        Returns:
            A tuple of the serialized rendezvous state and its fencing token.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """


class RendezvousTimeout:
    """Holds the timeout configuration of a rendezvous.

    Args:
        join:
            The total time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            minimum number of participants has been reached.
        close:
            The time within which the rendezvous is expected to close after a
            call to `set_closed()` or `shutdown()`.
        store:
            The timeout of the `Store` instance returned by the rendezvous.
    """

    # fmt: off

    _ZERO: Final = timedelta(0)

    _DEFAULT_TIMEOUTS: Final = {
        "join":      timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close":     timedelta(seconds=30),
        "store":     timedelta(seconds=300),
    }

    _join:      timedelta
    _last_call: timedelta
    _close:     timedelta
    _store:     timedelta

    def __init__(
        self,
        join:      Optional[timedelta] = None,
        last_call: Optional[timedelta] = None,
        close:     Optional[timedelta] = None,
        store:     Optional[timedelta] = None,
    ) -> None:
        self._set_timeouts(join=join, last_call=last_call, close=close, store=store)

    # fmt: on

    @property
    def join(self) -> timedelta:
        """Gets the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Gets the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Gets the close timeout."""
        return self._close

    @property
    def store(self) -> timedelta:
        """Gets the store timeout."""
        return self._store

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            else:
                if timeout <= self._ZERO:
                    raise ValueError(f"The {name} timeout must be a positive integer.")
            setattr(self, "_" + name, timeout)


@final
class DefaultRendezvousHandler(RendezvousHandler):
    """Represents the default rendezvous handler.

    Args:
        run_id:
            The id of the rendezvous.
        backend:
            The backend to use to hold the state of the rendezvous.
        min_num_participants:
            The minimum number of participants to admit to the rendezvous.
        max_num_participants:
            The maximum number of participants to admit to the rendezvous.
        store_port:
            The port number of the store returned by the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
    """

    # Static
    _id_generator: Final = _WorkerIdGenerator()

    _worker_id: _WorkerId
    _run_id: str
    _backend: RendezvousBackend
    _min_num_participants: int
    _max_num_participants: int
    _timeout: RendezvousTimeout
    _store: Optional[_ClosableStore]
    _store_port: int
    _rendezvous: _Rendezvous
    _rendezvous_token: Token
    _rendezvous_dirty: bool
    _last_sync: float
    _lock: threading.Lock
    _keep_alive_timer: Optional[_PeriodicTimer]
    _keep_alive_interval: timedelta
    _keep_alive_max_count: int
    _operation_deadline: float

    def __init__(
        self,
        run_id: str,
        backend: RendezvousBackend,
        min_num_participants: int,
        max_num_participants: int,
        store_port: int = 29500,
        timeout: Optional[RendezvousTimeout] = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        if min_num_participants < 1:
            raise ValueError("The minimum number of participants must be greater than zero.")

        if max_num_participants < min_num_participants:
            raise ValueError(
                "The maximum number of participants must be greater than or equal to the minimum number of participants."
            )

        if store_port >= 2 ** 16:
            raise ValueError("The port number of the store must be less than 65536.")

        self._worker_id = self._id_generator.generate()
        self._run_id = run_id

        self._backend = backend

        self._min_num_participants = min_num_participants
        self._max_num_participants = max_num_participants

        self._timeout = timeout or RendezvousTimeout()

        # Holds the store to be returned by the rendezvous.
        self._store = None
        self._store_port = store_port

        # Holds the rendezvous state synchronized with other workers via the
        # rendezvous backend.
        self._rendezvous = _Rendezvous()
        self._rendezvous_token = None
        self._rendezvous_dirty = False
        self._last_sync = 0.0

        self._lock = threading.Lock()

        self._keep_alive_timer = None
        self._keep_alive_interval = timedelta(seconds=10)
        self._keep_alive_max_count = 3

        self._operation_deadline = 0.0

    def __del__(self) -> None:
        self._disable_keep_alive_updates()

    @property
    def worker_id(self) -> str:
        """Returns the id of this worker."""
        return str(self._worker_id)

    @property
    def min_num_participants(self) -> int:
        """Gets the minimum number of participants to admit to the rendezvous."""
        return self._min_num_participants

    @property
    def max_num_participants(self) -> int:
        """Gets the maximum number of participants to admit to the rendezvous."""
        return self._max_num_participants

    @property
    def store_port(self) -> int:
        """Gets the port number of the store returned by the rendezvous."""
        return self._store_port

    @property
    def timeout(self) -> RendezvousTimeout:
        """Gets the timeout configuration of the rendezvous."""
        return self._timeout

    def get_backend(self) -> str:
        """See base class."""
        return self._backend.name

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        """See base class."""
        self._disable_keep_alive_updates()

        # Delay the execution for a small random amount of time if this is our
        # first round. This will slightly skew the rendezvous attempts across
        # the workers and reduce the load on the backend.
        if self._rendezvous is None:
            self._delay(seconds=(0, 0.2))

        self._set_operation_timeout(self._timeout.join)

        self._run(self._exit_rendezvous)
        self._run(self._join_rendezvous)

        self._enable_keep_alive_updates()

        return self._get_next_rendezvous_result()

    def is_closed(self) -> bool:
        """See base class."""
        with self._lock:
            self._sync_rendezvous()

            return self._rendezvous.closed

    def set_closed(self) -> None:
        """See base class."""
        with self._lock:
            self._set_operation_timeout(self._timeout.close)

            self._run(self._close_rendezvous)

    def num_nodes_waiting(self) -> int:
        """See base class."""
        with self._lock:
            self._sync_rendezvous()

            return len(self._rendezvous.wait_list)

    def get_run_id(self) -> str:
        """See base class."""
        return self._run_id

    def shutdown(self) -> bool:
        """See base class."""
        self._disable_keep_alive_updates()

        try:
            self.set_closed()

            return True
        except RendezvousError:
            return False

    def _get_next_rendezvous_result(self):
        rdzv = self._rendezvous

        rank, world_size = rdzv.participants[self._worker_id], len(rdzv.participants)

        is_store_host = rdzv.store_host == self._worker_id

        if self._store:
            self._store.close()

        self._store = TCPStore(
            rdzv.store_host.hostname,
            self._store_port,
            world_size,
            is_store_host,
            self._timeout.store,
        )

        return self._store, rank, world_size

    def _set_operation_timeout(self, timeout: timedelta) -> None:
        # We use a monotonic clock to avoid time drifts.
        self._operation_deadline = time.monotonic() + timeout.total_seconds()

    def _enable_keep_alive_updates(self) -> None:
        self._keep_alive_timer = _PeriodicTimer(self._keep_alive_interval, self._run_keep_alive)

        self._keep_alive_timer.start()

    def _disable_keep_alive_updates(self) -> None:
        if self._keep_alive_timer:
            self._keep_alive_timer.cancel()

    def _run_keep_alive(self) -> None:
        self._lock.acquire()

        try:
            self._set_operation_timeout(timedelta(seconds=2))

            self._run(self._keep_alive)
        except RendezvousError:
            pass
        finally:
            self._lock.release()

    def _run(self, state_handler: Callable[[], _Action]) -> None:
        action = None

        while action != _Action.FINISH:
            self._sync_rendezvous()

            # Determine the next action to take.
            action = state_handler()

            if action == _Action.FINISH:
                continue

            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError()

            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError()

            if action == _Action.SYNC_RENDEZVOUS:
                # Avoid overloading the backend if we are asked to poll for
                # state changes.
                self._delay(seconds=1)
            else:
                if action == _Action.UPDATE_KEEP_ALIVE:
                    self._update_keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()

                # Ensure that we synchronize our local changes to the backend.
                self._rendezvous_dirty = True

    def _sync_rendezvous(self) -> None:
        if self._rendezvous_dirty:
            state = pickle.dumps(self._rendezvous)

            response = self._backend.set_state(state, self._rendezvous_token)
        else:
            # Avoid overloading the backend if we are asked to retrieve the
            # state repeatedly. Serve the cached state for 1 second.
            if self._last_sync > time.monotonic() - 1:
                return

            response = self._backend.get_state()

        if response:
            state, token = response

            try:
                self._rendezvous = pickle.loads(state)
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See chained exception for details."
                ) from exc
        else:
            token = None

            self._rendezvous = _Rendezvous()

        self._rendezvous_token = token
        self._rendezvous_dirty = False

        self._last_sync = time.monotonic()

        self._sanitize_rendezvous()

    def _sanitize_rendezvous(self) -> None:
        pass

    #        rdzv = self._rendezvous
    #
    #        expire_time = self._utcnow - self._keep_alive_interval
    #
    #        alive_workers = {
    #            worker_id: last_keep_alive
    #                for worker_id, last_keep_alive in rdzv.last_keep_alives.items()  # noqa E131
    #                    if last_keep_alive >= expire_time                            # noqa E131
    #        }
    #
    #        # No "dead" workers, skip the rest.
    #        if len(alive_workers) == len(rdzv.last_keep_alives):
    #            return
    #
    #        rdzv.participants = {
    #            worker_id: rank
    #                for worker_id, rank in participants.items()  # noqa E131
    #                    if worker_id in alive_workers            # noqa E131
    #        }
    #
    #        rdzv.wait_list = {
    #            worker_id for worker_id in rdzv.wait_list if worker_id in alive_workers
    #        }
    #
    #        rdzv.last_keep_alives = alive_workers
    #
    @staticmethod
    def _delay(seconds: Union[float, Tuple[float, float]]) -> None:
        if isinstance(seconds, tuple):
            seconds = random.uniform(*seconds)
        # Ignore delay requests that are less than 10 milliseconds.
        if seconds >= 0.01:
            time.sleep(seconds)

    def _join_rendezvous(self) -> _Action:
        rdzv = self._rendezvous

        is_participant = self._worker_id in rdzv.participants

        # If we are part of the rendezvous and it is already complete there is
        # no further action to take. We can finish the call.
        if rdzv.complete and is_participant:
            return _Action.FINISH

        # A closed rendezvous means that it no longer accepts new workers.
        if rdzv.closed:
            return _Action.ERROR_CLOSED

        now = time.monotonic()
        if now > self._operation_deadline:
            # If we still have time to rollback, try to remove ourself from the
            # rendezvous. This will speed up things, but it is okay if we can't
            # as our keep-alive will eventually expire.
            if now <= self._operation_deadline + 5:
                # If we are part of the rendezvous, it means we couldn't find
                # enough participants to execute it on time.
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                # If we are in the wait list, it means we couldn't wait till the
                # next round of the rendezvous.
                if self._worker_id in rdzv.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT

        if rdzv.complete:
            # If we are here, it means we are not part of the rendezvous. In
            # case the rendezvous has capacity for additional participants add
            # ourself to the wait list for the next round.
            if len(rdzv.participants) < self._max_num_participants:
                if not self._worker_id in rdzv.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
        elif is_participant:
            # If the rendezvous has enough number of participants including us,
            # check whether we have passed the rendezvous deadline. If yes,
            # complete it.
            if len(rdzv.participants) >= self._min_num_participants:
                if cast(datetime, rdzv.deadline) > datetime.utcnow():
                    return _Action.MARK_RENDEZVOUS_COMPLETE
        else:
            # The rendezvous is not complete yet and we are not part of it. Try
            # to join.
            return _Action.ADD_TO_PARTICIPANTS

        if self._should_keep_alive():
            return _Action.UPDATE_KEEP_ALIVE

        # At this point either the rendezvous is not complete, but we are part
        # of it, which means we have to wait for other participants to join; or
        # the rendezvous is complete, but we are not part of it, which means we
        # have to wait for the next round.
        return _Action.SYNC_RENDEZVOUS

    def _exit_rendezvous(self) -> _Action:
        if self._worker_id in self._rendezvous.participants:
            if time.monotonic() > self._operation_deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH

    def _keep_alive(self) -> _Action:
        if self._should_keep_alive():
            if time.monotonic() > self._operation_deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.UPDATE_KEEP_ALIVE
        return _Action.FINISH

    def _close_rendezvous(self) -> _Action:
        if self._rendezvous.closed:
            return _Action.FINISH
        if time.monotonic() > self._operation_deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED

    def _should_keep_alive(self) -> bool:
        try:
            last_keep_alive = self._rendezvous.last_keep_alives[self._worker_id]
        except KeyError:
            return False

        return last_keep_alive <= datetime.utcnow() - self._keep_alive_interval

    def _update_keep_alive(self) -> None:
        self._rendezvous.last_keep_alives[self._worker_id] = datetime.utcnow()

    def _add_to_participants(self) -> None:
        rdzv = self._rendezvous

        try:
            rdzv.wait_list.remove(self._worker_id)
        except KeyError:
            pass

        # The ranks of the participants will be set once the rendezvous is
        # complete.
        rdzv.participants[self._worker_id] = 0

        self._update_keep_alive()

        if len(rdzv.participants) == self._min_num_participants:
            # Start the last call timer.
            rdzv.deadline = datetime.utcnow() + self._timeout.last_call

        if len(rdzv.participants) == self._max_num_participants:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        self._rendezvous.wait_list.add(self._worker_id)

        self._update_keep_alive()

    def _remove_from_participants(self) -> None:
        rdzv = self._rendezvous

        del rdzv.participants[self._worker_id]

        del rdzv.last_keep_alives[self._worker_id]

        # If we do not have any participants left, move to the next round.
        if not rdzv.participants:
            rdzv.complete, rdzv.store_host = False, None

            rdzv.round += 1

    def _remove_from_wait_list(self) -> None:
        self._rendezvous.wait_list.remove(self._worker_id)

        del self._rendezvous.last_keep_alives[self._worker_id]

    def _mark_rendezvous_complete(self) -> None:
        rdzv = self._rendezvous

        rdzv.complete, rdzv.deadline = True, None

        # Assign the ranks.
        for rank, worker_id in enumerate(rdzv.participants):
            rdzv.participants[worker_id] = rank

            if rank == 0:
                rdzv.store_host = worker_id

    def _mark_rendezvous_closed(self) -> None:
        self._rendezvous.closed = True


def _parse_store_port(params: RendezvousParameters) -> Optional[int]:
    port_str = params.config.get("store_port")
    if port_str is None:
        return None

    port = _try_parse_port(port_str)
    if port is None:
        raise ValueError("The store port number must be an integer between 0 and 65536.")
    return port


def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    backend: RendezvousBackend, params: RendezvousParameters
) -> DefaultRendezvousHandler:
    """Create a new `DefaultRendezvousHandler` from the specified parameters."""
    kwargs: Dict[str, Any] = {}

    store_port = _parse_store_port(params)
    if store_port:
        kwargs["store_port"] = store_port

    kwargs["timeout"] = RendezvousTimeout(
        _get_timeout(params, "join"),
        _get_timeout(params, "last_call"),
        _get_timeout(params, "close"),
        _get_timeout(params, "store"),
    )

    return DefaultRendezvousHandler(
        params.run_id,
        backend,
        params.min_nodes,
        params.max_nodes,
        **kwargs,
    )
