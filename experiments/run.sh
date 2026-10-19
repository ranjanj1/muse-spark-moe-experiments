#!/bin/bash
# Run one experiment and keep a replayable record: experiments/logs/<name>.log holds the command, start time, and every
# output line with seconds since start (for an on-screen terminal replay). Usage: experiments/run.sh <name> <command...>
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
name=$1; shift
log=experiments/logs/$name.log
set -a; source ../../.env >/dev/null 2>&1; set +a     # HF_TOKEN; never echoed
{ echo "# $(date '+%Y-%m-%d %H:%M:%S')  \$ $*"; } > "$log"
start=$(python3 -c 'import time;print(time.time())')
"$@" 2>&1 | grep --line-buffered -v "Redirects are currently not supported" | while IFS= read -r line; do
  printf '%7.1f  %s\n' "$(python3 -c "import time;print(time.time()-$start)")" "$line"
done | tee -a "$log"
echo "# exit ${PIPESTATUS[0]}" >> "$log"
