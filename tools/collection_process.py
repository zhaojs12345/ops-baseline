# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded collector subprocesses without changing enclosing process groups."""
import os
import signal
import subprocess


def _kill_tree(root):
    """Freeze parents before discovering children, then kill leaves first.

    Keeping the inherited process group lets an outer scheduler still cancel the
    entire operator. Freezing prevents a launcher from forking another child
    between enumeration and cleanup. Only the timed-out command's tree is touched.
    """
    stopped = []
    pending = {root}
    try:
        while pending:
            for pid in pending:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                stopped.append(pid)
            table = subprocess.check_output(['ps', '-eo', 'pid=,ppid='],
                                            text=True, timeout=5)
            parents = set(stopped)
            pending = {int(pid) for line in table.splitlines()
                       for pid, ppid in [line.split()]
                       if int(ppid) in parents and int(pid) not in parents}
    finally:
        # Even a failed process-table query must not leave our launcher stopped.
        for pid in reversed(stopped):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_with_cleanup(command, *, timeout, capture_output=False, text=False):
    """Match collector run() calls; timeout/exception cleans live descendants."""
    pipe = subprocess.PIPE if capture_output else None
    with subprocess.Popen(command, stdout=pipe, stderr=pipe, text=text) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            try:
                _kill_tree(process.pid)
            finally:
                process.wait()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
