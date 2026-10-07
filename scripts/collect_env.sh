#!/usr/bin/env bash
# Usage: scripts/collect_env.sh > results/raw/<member>-<YYYYMMDD-HHMM>/environment.txt
set -u
cd "$(dirname "$0")/.."

show() {
    printf '\n$ %s\n' "$*"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1
    else
        echo "(not found)"
    fi
}

echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "worktree: dirty"
    git status --short
else
    echo "worktree: clean"
fi

show uname -a
[ -r /etc/os-release ] && show cat /etc/os-release
show lscpu
show free -h
show nvidia-smi
show nvcc --version
show "${CC:-cc}" --version
show python3 --version
show python3 -c "import numpy; print('numpy', numpy.__version__)"
show make --no-print-directory print-flags
