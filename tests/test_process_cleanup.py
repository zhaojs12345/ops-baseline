# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for real collector timeout call sites and shell launchers."""
import ast
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from collection_process import run_with_cleanup


def alive(pid):
    result = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
    state = result.stdout.strip()
    return bool(state) and not state.startswith('Z')


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pidfile = self.root / 'descendant.pid'
        self.leaf = self.root / 'leaf.sh'
        self.leaf.write_text('sleep 30 &\nprintf "%s" "$!" > "$PIDFILE"\nwait\n')
        self.wrapper = self.root / 'ncu'
        self.wrapper.write_text('#!/bin/sh\n/bin/sh "$LEAF" &\nwait\n')
        self.wrapper.chmod(0o700)
        self.environment = patch.dict(os.environ, {
            'PATH': str(self.root) + os.pathsep + os.environ['PATH'],
            'PIDFILE': str(self.pidfile), 'LEAF': str(self.leaf)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.kill_fixture)

    def kill_fixture(self):
        if self.pidfile.exists():
            try:
                os.kill(int(self.pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def assert_descendant_stopped(self):
        self.assertTrue(self.pidfile.exists(), 'fixture must start before timeout')
        pid = int(self.pidfile.read_text())
        deadline = time.monotonic() + 2
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertFalse(alive(pid), 'descendant survived collector timeout')

    def load_function(self, name):
        # Exercise actual collector functions without importing GPU dependencies.
        tree = ast.parse((TOOLS / 'collect_baseline_nvidia.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        def bounded(run):
            def call(command, **kwargs):
                kwargs['timeout'] = 1
                return run(command, **kwargs)
            return call
        namespace = dict(Path=Path, os=os, sys=sys, json=json, tempfile=tempfile,
                         __file__=str(self.root / 'worker.py'), DEFAULT_REPO='pytorch',
                         FLOP_TENSOR_CORE={}, FLOP_CUDA_CORE={}, MEMORY_BYTES='mem',
                         UTIL={}, SM_UTIL='sm', DURATION='duration', math=math,
                         subprocess=SimpleNamespace(run=bounded(subprocess.run),
                                                    TimeoutExpired=subprocess.TimeoutExpired),
                         run_with_cleanup=bounded(run_with_cleanup),
                         _empty_result=lambda *args, **kwargs: {'num_kernels': 0})
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<collector>', 'exec'), namespace)
        return namespace[name]

    def test_ncu_export_timeout_stops_launcher_child(self):
        profile = self.load_function('profile_with_ncu')
        with redirect_stdout(io.StringIO()):
            result = profile('test', '', [1], 'torch.float32', report_dir=self.root)
        self.assertEqual(result['num_kernels'], 0)
        self.assert_descendant_stopped()

    def test_ncu_import_timeout_stops_launcher_child(self):
        self.wrapper.write_text('#!/bin/sh\nif [ "$1" = "--import" ]; then\n'
                                '  /bin/sh "$LEAF" &\n  wait\nelse\n'
                                '  while [ "$1" != "--export" ]; do shift; done\n'
                                '  shift\n  touch "$1.ncu-rep"\nfi\n')
        profile = self.load_function('profile_with_ncu')
        with self.assertRaises(subprocess.TimeoutExpired):
            profile('test', '', [1], 'torch.float32', report_dir=self.root)
        self.assert_descendant_stopped()

    def test_success_and_nonzero_exit_preserve_output_and_process_group(self):
        for code in (0, 7):
            script = f'import os,sys; print(os.getpgrp()); print("diagnostic",file=sys.stderr); sys.exit({code})'
            result = run_with_cleanup([sys.executable, '-c', script], timeout=5,
                                      capture_output=True, text=True)
            self.assertEqual(result.returncode, code)
            self.assertEqual(int(result.stdout), os.getpgrp())
            self.assertEqual(result.stderr.strip(), 'diagnostic')

    def test_missing_executable_still_raises(self):
        with self.assertRaises(FileNotFoundError):
            run_with_cleanup([str(self.root / 'missing')], timeout=1)

    def test_interruption_stops_descendants(self):
        communicate = subprocess.Popen.communicate
        def interrupted(process, *args, **kwargs):
            if str(self.wrapper) not in process.args:
                return communicate(process, *args, **kwargs)
            deadline = time.monotonic() + 5
            while not self.pidfile.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.pidfile.exists())
            raise KeyboardInterrupt
        with patch.object(subprocess.Popen, 'communicate', autospec=True, side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                run_with_cleanup([str(self.wrapper)], timeout=5, capture_output=True)
        self.assert_descendant_stopped()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux child subreaper regression')
    def test_exited_launcher_descendant_is_stopped(self):
        self.wrapper.write_text('#!/bin/sh\n/bin/sh "$LEAF" &\nexit 0\n')
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            run_with_cleanup([str(self.wrapper)], timeout=1, capture_output=True)
        self.assertEqual(caught.exception.cmd, [str(self.wrapper)])
        self.assert_descendant_stopped()

    def test_cleanup_does_not_require_ps_on_path(self):
        self.leaf.write_text('/bin/sleep 30 &\nprintf "%s" "$!" > "$PIDFILE"\nwait\n')
        with patch.dict(os.environ, {'PATH': str(self.root)}):
            with self.assertRaises(subprocess.TimeoutExpired):
                run_with_cleanup([str(self.wrapper)], timeout=1, capture_output=True)
        self.assert_descendant_stopped()

    def test_signal_returncode_and_binary_output(self):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=sig):
                code = ('import os,signal; os.write(1,b"out"); os.write(2,b"err"); '
                        f'os.kill(os.getpid(),{int(sig)})')
                result = run_with_cleanup([sys.executable, '-c', code], timeout=5, capture_output=True)
                self.assertEqual(result.returncode, -sig)
                self.assertEqual(result.stdout, b'out')
                self.assertEqual(result.stderr, b'err')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux child subreaper regression')
    def test_real_group_sigint_keeps_adoption_boundary_until_cleanup(self):
        # The target ignores Ctrl-C; the shell may exit first. The supervisor
        # must survive until the interrupted collector has killed its subtree.
        target = self.root / 'target.py'
        target.write_text('import os,signal,time\n'
                          'signal.signal(signal.SIGINT,signal.SIG_IGN)\n'
                          'open(os.environ["PIDFILE"],"w").write(str(os.getpid()))\n'
                          'time.sleep(30)\n')
        self.wrapper.write_text(f'#!/bin/sh\n"{sys.executable}" "{target}" &\nwait\n')
        code = ('from collection_process import run_with_cleanup\n'
                'try:\n'
                f'    run_with_cleanup([{str(self.wrapper)!r}],timeout=30,capture_output=True)\n'
                'except KeyboardInterrupt:\n'
                '    pass\n')
        worker = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                  env={**os.environ, 'PYTHONPATH': str(TOOLS)},
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not self.pidfile.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.pidfile.exists())
            os.killpg(worker.pid, signal.SIGINT)
            stdout, stderr = worker.communicate(timeout=5)
            self.assertEqual(worker.returncode, 0, stderr)
            self.assertEqual(stdout, b'')
            self.assertEqual(stderr, b'')
            self.assert_descendant_stopped()
        finally:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()

    def test_outer_process_group_cancellation_stops_command_descendants(self):
        # Same cancellation mechanism as the multi-GPU scheduler. Its SIGKILL
        # must reach the supervisor, command and descendants, with no forwarding.
        code = ('from collection_process import run_with_cleanup; '
                f'run_with_cleanup([{str(self.wrapper)!r}],timeout=30,capture_output=True)')
        worker = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                  env={**os.environ, 'PYTHONPATH': str(TOOLS)})
        try:
            deadline = time.monotonic() + 5
            while not self.pidfile.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.pidfile.exists())
        finally:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
        self.assert_descendant_stopped()

    def test_operator_timeout_preserves_partial_and_stops_descendants(self):
        partial = {'shapes': {'[1]': {'torch.float32': {'T_us': 1}}}}
        (self.root / 'worker.py').write_text(
            'import json,sys,subprocess,os\n'
            'out=sys.argv[sys.argv.index("--output")+1]\n'
            f'open(out,"w").write({json.dumps({"test": partial})!r})\n'
            'subprocess.call(["/bin/sh",os.environ["LEAF"]])\n')
        spawn = self.load_function('_spawn_op_worker')
        with redirect_stdout(io.StringIO()):
            entry, ok = spawn('test', True, self.root, None, 1)
        self.assertFalse(ok)
        self.assertEqual(entry, partial)
        self.assert_descendant_stopped()


if __name__ == '__main__':
    unittest.main()
