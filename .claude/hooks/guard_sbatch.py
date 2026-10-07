#!/usr/bin/env python
"""PreToolUse hook: refuse `sbatch` for training/sampling unless the run has a
planned ledger entry, and refuse destructive commands outright.

Claude Code passes the tool call as JSON on stdin. Exit 2 blocks the call and
sends stderr back to the model as the reason.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

data = json.load(sys.stdin)
if data.get("tool_name") != "Bash":
    sys.exit(0)
cmd = data.get("tool_input", {}).get("command", "")

# Hard blocks.
for pat, why in [
    (r"\brm\s+-rf?\b|\brm\s+", "rm is not allowed from the agent; ask the user to delete"),
    (r"\bscancel\b", "scancel needs APPROVE DELETE from the user"),
    (r"\bpip\s+install\b|\bconda\s+install\b", "no installs or env changes (CLAUDE.md)"),
    (r"--overwrite\b", "never overwrite existing samples or results"),
    (r"\bgit\s+push\b", "git push needs APPROVE PUSH from the user"),
]:
    if re.search(pat, cmd):
        print(f"blocked: {why}", file=sys.stderr)
        sys.exit(2)

# sbatch: allowed only for a run with a planned ledger entry, or for CPU
# analysis jobs on the standard partition that name no run.
if re.search(r"\bsbatch\b", cmd):
    m = re.search(r"--export=.*?RUN_NAME=([A-Za-z0-9_.\-]+)", cmd) or re.search(r"RUN_NAME=([A-Za-z0-9_.\-]+)", cmd)
    if not m:
        print("blocked: sbatch must carry RUN_NAME=<run> (via --export) so the ledger can be checked; "
              "CPU analysis jobs may use RUN_NAME=analysis", file=sys.stderr)
        sys.exit(2)
    run = m.group(1)
    if run == "analysis":
        sys.exit(0)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ledger.py"), "check-planned", "--run", run],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"blocked: run {run} has no planned ledger entry ({r.stdout.strip()}). "
              f"Have the planner add one first.", file=sys.stderr)
        sys.exit(2)
sys.exit(0)
