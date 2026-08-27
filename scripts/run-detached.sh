#!/usr/bin/env bash
# Run a long job in its own session, detached from this terminal/agent session,
# so that closing the session (or its cleanup) cannot kill it. Output is
# appended to LOGFILE. macOS has no setsid(1), so the detach is done in Python.
#
#   scripts/run-detached.sh dist/census-r6.log ./venv/bin/python scripts/populate.py ...
#   tail -f dist/census-r6.log
#   pkill -f 'bigcell'                       # to stop it
#
# `caffeinate -i` is applied automatically so the machine will not idle-sleep
# while the job runs (a closed lid still sleeps; keep the lid open).
set -euo pipefail
LOG=${1:?usage: run-detached.sh LOGFILE COMMAND...}; shift
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"
exec /usr/bin/python3 -c '
import os, sys
log, cmd = sys.argv[1], ["caffeinate", "-i", *sys.argv[2:]]
if os.fork():                      # parent: report and return to the shell
    sys.exit(0)
os.setsid()                        # child: new session, no controlling terminal
if os.fork():
    os._exit(0)                    # grandchild does the work
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
' "$LOG" "$@"
