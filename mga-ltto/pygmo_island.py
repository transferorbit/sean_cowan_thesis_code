'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 03-08-2022

This module defines an island class that will be used to perform parallel optimization processes
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
import pygmo as pg

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory

import mga_low_thrust_utilities as mga_util
from threading import Lock as _Lock
from multiprocessing import Pool

class my_isl:
    def run_evolve(self, algo, pop):
        new_pop = algo.evolve(pop)
        return algo, new_pop

    def get_name(self):
        return "It's my island!"


def _evolve_func(algo, pop): # doctest : +SKIP
    new_pop = algo.evolve(pop)
    return algo, new_pop

class userdefinedisland_v2(object): # doctest : +SKIP

    def __init__(self):
        # Init the process pool, if necessary.
        mp_island.init_pool()

    def run_evolve(self, algo, pop):
        with mp_island._pool_lock:
            res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
        return res.get()

class userdefinedisland_v3: #THIS WORKS

    def run_evolve(self, algo, pop):
        new_pop = algo.evolve(pop)
        return algo, new_pop

    def get_name(self):
        return "It's my island!"

class userdefinedisland_v4:

    def run_evolve(self, algo, pop):
        if __name__ == '__main__':
            with Pool(processes=8) as pool:
                result = pool.map_async(_evolve_func, (algo, pop))
                return result.get()

    def get_name(self):
        return "It's my island!"

class userdefinedisland_test:

    def run_evolve(self):
        def f(x):
            for i in range(x):
                for j in range(x):
                    for k in range(x):
                        x +=3
            return x*x
        if __name__ == '__main__':
            with Pool(processes=4) as pool:         # start 4 worker processes
                result = pool.map_async(f, range(10))       # prints "[0, 1, 4,..., 81]"
                return result.get()

class userDefinedIsland:
    """
    """

    def __init__(self) -> None:
        pass

    def run_evolve(self, algo, pop):
        # NOTE: the idea here is that we pass the *already serialized*
        # arguments to the mp machinery, instead of letting the multiprocessing
        # module do the serialization. The advantage of doing so is
        # that if there are serialization errors, we catch them early here rather
        # than failing in the bootstrap phase of the remote process, which
        # can lead to hangups.
        import pickle
        ser_algo_pop = pickle.dumps((algo, pop))

        if self._use_pool:
            with mp_island._pool_lock:
                # NOTE: run this while the pool is locked. We have
                # functions to modify the pool (e.g., resize()) and
                # we need to make sure we are not trying to touch
                # the pool while we are sending tasks to it.
                if mp_island._pool is None:
                    raise RuntimeError(
                        "The multiprocessing island pool was stopped. Please restart it via mp_island.init_pool().")
                res = mp_island._pool.apply_async(
                    _evolve_func_mp_pool, (ser_algo_pop,))
            # NOTE: there might be a bug in need of a workaround lurking in here:
            # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
            # Just keep it in mind.
            return pickle.loads(res.get())
        else:

            # Get the context for spawning the process.
            mp_ctx = _get_spawn_context()

            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            p = mp_ctx.Process(target=_evolve_func_mp_pipe,
                               args=(child_conn, ser_algo_pop))
            p.start()
            with self._pid_lock:
                self._pid = p.pid
            # NOTE: after setting the pid, wrap everything
            # in a try block with a finally clause for
            # resetting the pid to None. This way, even
            # if there are exceptions, we are sure the pid
            # is set back to None.
            try:
                res = parent_conn.recv()
                p.join()
            finally:
                with self._pid_lock:
                    self._pid = None
            if isinstance(res, RuntimeError):
                raise res
            return pickle.loads(res)




class mp_island(object):
    # Static variables for the pool.
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self, use_pool=True):
        """
        Args:

           use_pool(:class:`bool`): if :data:`True`, a process from a global pool will be used to run the evolution, otherwise a new
              process will be spawned for each evolution

        Raises:

           TypeError: if *use_pool* is not of type :class:`bool`
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()` if *use_pool* is :data:`True`

        """
        self._init(use_pool)

    def _init(self, use_pool):
        # Implementation of the ctor. Factored out
        # because it's re-used in the pickling support.
        if not isinstance(use_pool, bool):
            raise TypeError(
                "The 'use_pool' parameter in the mp_island constructor must be a boolean, but it is of type {} instead.".format(type(use_pool)))
        self._use_pool = use_pool
        if self._use_pool:
            # Init the process pool, if necessary.
            mp_island.init_pool()
        else:
            # Init the pid member and associated lock.
            self._pid_lock = _Lock()
            self._pid = None

    @property
    def use_pool(self):
        """Pool usage flag (read-only).

        Returns:

           :class:`bool`: :data:`True` if this island uses a process pool, :data:`False` otherwise

        """
        return self._use_pool

    def __copy__(self):
        # For copy/deepcopy, construct a new instance
        # with the same arguments used to construct self.
        # NOTE: no need for locking, as _use_pool is set
        # on construction and never touched again.
        return mp_island(self._use_pool)

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        # For pickle/unpickle, we employ the construction
        # argument, which will be used to re-init the class
        # during unpickle.
        return self._use_pool

    def __setstate__(self, state):
        # NOTE: we need to do a full init of the object,
        # in order to set the use_pool flag and, if necessary,
        # construct the _pid and _pid_lock objects.
        self._init(state)

    def run_evolve(self, algo, pop):
        # NOTE: the idea here is that we pass the *already serialized*
        # arguments to the mp machinery, instead of letting the multiprocessing
        # module do the serialization. The advantage of doing so is
        # that if there are serialization errors, we catch them early here rather
        # than failing in the bootstrap phase of the remote process, which
        # can lead to hangups.
        import pickle
        ser_algo_pop = pickle.dumps((algo, pop))

        if self._use_pool:
            with mp_island._pool_lock:
                # NOTE: run this while the pool is locked. We have
                # functions to modify the pool (e.g., resize()) and
                # we need to make sure we are not trying to touch
                # the pool while we are sending tasks to it.
                if mp_island._pool is None:
                    raise RuntimeError(
                        "The multiprocessing island pool was stopped. Please restart it via mp_island.init_pool().")
                res = mp_island._pool.apply_async(
                    _evolve_func_mp_pool, (ser_algo_pop,))
            # NOTE: there might be a bug in need of a workaround lurking in here:
            # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
            # Just keep it in mind.
            return pickle.loads(res.get())
        else:

            # Get the context for spawning the process.
            mp_ctx = _get_spawn_context()

            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            p = mp_ctx.Process(target=_evolve_func_mp_pipe,
                               args=(child_conn, ser_algo_pop))
            p.start()
            with self._pid_lock:
                self._pid = p.pid
            # NOTE: after setting the pid, wrap everything
            # in a try block with a finally clause for
            # resetting the pid to None. This way, even
            # if there are exceptions, we are sure the pid
            # is set back to None.
            try:
                res = parent_conn.recv()
                p.join()
            finally:
                with self._pid_lock:
                    self._pid = None
            if isinstance(res, RuntimeError):
                raise res
            return pickle.loads(res)

    @property
    def pid(self):
        """ID of the evolution process (read-only).

        This property is available only if the island is *not* using a process pool.

        Returns:

           :class:`int`: the ID of the process running the current evolution, or :data:`None` if no evolution is ongoing

        Raises:

           ValueError: if the island is using a process pool

        """
        if self._use_pool:
            raise ValueError(
                "The 'pid' property is available only when the island is configured to spawn new processes, but this mp_island is using a process pool instead.")
        with self._pid_lock:
            pid = self._pid
        return pid

    def get_name(self):
        """Island's name.

        Returns:

           :class:`str`: ``"Multiprocessing island"``

        """
        return "Multiprocessing island"

    def get_extra_info(self):
        """Island's extra info.

        If the island uses a process pool and the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`,
        invoking this function will trigger the creation of a new pool.

        Returns:

           :class:`str`: a string containing information about the state of the island (e.g., number of processes in the pool, ID of the evolution process, etc.)

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.get_pool_size()`

        """
        retval = "\tUsing a process pool: {}\n".format(
            "yes" if self._use_pool else "no")
        if self._use_pool:
            retval += "\tNumber of processes in the pool: {}".format(
                mp_island.get_pool_size())
        else:
            with self._pid_lock:
                pid = self._pid
            if pid is None:
                retval += "\tNo active evolution process"
            else:
                retval += "\tEvolution process ID: {}".format(pid)
        return retval

    @staticmethod
    def _init_pool_impl(processes):
        # Implementation method for initing
        # the pool. This will *not* do any locking.

        if mp_island._pool is None:
            mp_island._pool, mp_island._pool_size = _make_pool(processes)

    @staticmethod
    def init_pool(processes=None):
        """Initialise the process pool.

        This method will initialise the process pool backing :class:`~pygmo.mp_island`, if the pool
        has not been initialised yet or if the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`.
        Otherwise, this method will have no effects.

        Args:

           processes(:data:`None` or an :class:`int`): the size of the pool (if :data:`None`, the size of the pool will be
             equal to the number of logical CPUs on the system)

        Raises:

           ValueError: if the pool does not exist yet and the function is being called from a thread different
             from the main one, or if *processes* is a non-positive value
           TypeError: if *processes* is not :data:`None` and not an :class:`int`

        """
        with mp_island._pool_lock:
            mp_island._init_pool_impl(processes)

    @staticmethod
    def get_pool_size():
        """Get the size of the process pool.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Returns:

           :class:`int`: the current size of the pool

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        with mp_island._pool_lock:
            mp_island._init_pool_impl(None)
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        """Resize pool.

        This method will resize the process pool backing :class:`~pygmo.mp_island`.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Args:

           processes(:class:`int`): the desired number of processes in the pool

        Raises:

           TypeError: if the *processes* argument is not an :class:`int`
           ValueError: if the *processes* argument is not strictly positive
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """

        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError(
                "The 'processes' argument must be strictly positive")

        with mp_island._pool_lock:
            # NOTE: this will either init a new pool
            # with the requested number of processes,
            # or do nothing if the pool exists already.
            mp_island._init_pool_impl(processes)
            if processes == mp_island._pool_size:
                # Don't do anything if we are not changing
                # the size of the pool.
                return
            # Create new pool.
            new_pool, new_size = _make_pool(processes)
            # Stop the current pool.
            mp_island._pool.close()
            mp_island._pool.join()
            # Assign the new pool.
            mp_island._pool = new_pool
            mp_island._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        """Shutdown pool.

        .. versionadded:: 2.8

        This method will shut down the process pool backing :class:`~pygmo.mp_island`, after
        all pending tasks in the pool have completed.

        After the process pool has been shut down, attempting to run an evolution on the island
        will raise an error. A new process pool can be created via an explicit call to
        :func:`~pygmo.mp_island.init_pool()` or one of the methods of the public API of
        :class:`~pygmo.mp_island` which trigger the creation of a new process pool.

        """
        with mp_island._pool_lock:
            if mp_island._pool is not None:
                mp_island._pool.close()
                mp_island._pool.join()
                mp_island._pool = None
                mp_island._pool_size = None


# Copyright 2020, 2021 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


def _get_spawn_context():
    # Small utlity to get a context that will use the 'spawn' method to
    # create new processes with the multiprocessing module. We want to enforce
    # a uniform way of creating new processes across platforms in
    # order to prevent users from implicitly relying on platform-specific
    # behaviour (e.g., fork()), only to discover later that their
    # code is not portable across platforms. See:
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    import multiprocessing as mp

    return mp.get_context('spawn')


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C temporarily.

    def __enter__(self):
        import signal
        # Store the previous sigint handler and assign the new sig handler
        # (i.e., ignore SIGINT).
        self._prev_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal
        # Restore the previous sighandler.
        signal.signal(signal.SIGINT, self._prev_signal)


def _make_pool(processes):
    # A small factory function to create a process pool.
    # It accomplishes the tasks of selecting the correct method for
    # starting the processes ("spawn") and making sure that the
    # created processes will ignore the SIGINT signal (this prevents
    # troubles when the user issues an interruption with ctrl+c from
    # the main process).

    if processes is not None and not isinstance(processes, int):
        raise TypeError("The 'processes' argument must be None or an int")

    if processes is not None and processes <= 0:
        raise ValueError(
            "The 'processes' argument, if not None, must be strictly positive")

    # Get the context for spawning the process.
    mp_ctx = _get_spawn_context()

    # NOTE: we temporarily disable sigint while creating the pool.
    # This ensures that the processes created in the pool will ignore
    # interruptions issued via ctrl+c (only the main process will
    # be affected by them).
    with _temp_disable_sigint():
        pool = mp_ctx.Pool(processes=processes)

    pool_size = mp_ctx.cpu_count() if processes is None else processes

    # Return the created pool and its size.
    return pool, pool_size


def _evolve_func_mp_pool(ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when using the pool).
    import pickle
    algo, pop = pickle.loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return pickle.dumps((algo, new_pop))


def _evolve_func_mp_pipe(conn, ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when *not* using the pool). Communication with the
    # parent process happens through the conn pipe.

    # NOTE: disable SIGINT with the goal of preventing the user from accidentally
    # interrupting the evolution via hitting Ctrl+C in an interactive session
    # in the parent process. Note that this disables the signal only during
    # evolution, but the signal is still enabled when the process is bootstrapping
    # (so the user can still cause troubles in the child process with a well-timed
    # Ctrl+C). There's nothing we can do about it: the only way would be to disable
    # SIGINT before creating the child process, but unfortunately the creation
    # of a child process happens in a separate thread and Python disallows messing
    # with signal handlers from a thread different from the main one :(
    with _temp_disable_sigint():
        import pickle
        try:
            algo, pop = pickle.loads(ser_algo_pop)
            new_pop = algo.evolve(pop)
            conn.send(pickle.dumps((algo, new_pop)))
        except Exception as e:
            conn.send(RuntimeError(
                "An exception was raised in the evolution of a multiprocessing island. The full error message is:\n{}".format(e)))
        finally:
            conn.close()
