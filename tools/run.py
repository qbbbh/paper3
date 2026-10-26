"""Dispatch to byte-identical frozen scripts; this wrapper does not select an experiment."""
from pathlib import Path
import subprocess
import sys
from build_runtime import build

commands = {'allocation':'run_allocation.py', 'foundation':'run_foundation.py',
            'crlb':'crlb_report.py', 'estimation_plot':'plot_estimation.py',
            'audit':'audit_allocation_outputs.py'}
if len(sys.argv) < 2 or sys.argv[1] not in {*commands, 'test', 'prepare'}:
    raise SystemExit('Usage: python tools/run.py {prepare|test|audit|allocation|foundation|crlb|estimation_plot} [original arguments]')
runtime = build()
task = sys.argv[1]
args = sys.argv[2:]
if task == 'prepare':
    print(runtime)
    raise SystemExit(0)
if task == 'test':
    command = [sys.executable, '-m', 'pytest', *args]
    cwd = runtime / 'code'
else:
    command = [sys.executable, str(runtime / 'code' / commands[task]), *args]
    # Keep explicit relative output/data paths relative to the caller, not an invisible cache.
    cwd = Path.cwd()
raise SystemExit(subprocess.call(command, cwd=cwd))
