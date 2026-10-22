#!/bin/bash -l
# Runs on the self-hosted runner (GPU node): job helicity of gpu_runner_ci.yml.
# The branch-free helicities and the single-helicity evaluation on the GPU backend of
# madmatrix: madmatrix_helicity_checks.py (the same results as the reference MadGraph7, the
# polarised |M|^2 add up to the unpolarised one, against the scalar CPU backend too) and
# madmatrix_helicity_bench.py (runtime of the wavefunction routines and of the whole |M|^2
# against the reference), see those for what is done.
# Environment: as madmatrix_gpu.sh (BACKEND, MODULES, GPU_ARCH, VENV, WORKDIR), and
# REFERENCE_REF, the commit of the reference MadGraph7 (default: main before the branch-free
# helicities).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
REFERENCE_REF=${REFERENCE_REF:-6b7fba490b2d8ea9b7fb0fc6860c8204d190a24e}
section() { echo; echo "=================== $* ($(date +%T))"; }

section "Environment"
if [ -n "$MODULES" ]; then source "$HERE/load_modules.sh"; fi
source "$VENV/bin/activate"
python3 --version
case $BACKEND in
    cuda) nvidia-smi
          GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p) ;;
    hip)  rocm-smi || true
          GPU_NAME=$(rocminfo 2> /dev/null | awk -F': *' '/Marketing Name/ {name=$2} /Device Type: *GPU/ {print name; exit}' || true) ;;
    *)    echo "::error::the helicity checks need a GPU backend, not '$BACKEND'"; exit 2 ;;
esac
export GPU_NAME
if [ -n "$GPU_ARCH" ]; then
    export MADGRAPH_CUDA_ARCHITECTURE=$GPU_ARCH MADGRAPH_HIP_ARCHITECTURE=$GPU_ARCH
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/reference"

section "Reference MadGraph7 ($REFERENCE_REF)"
# the checkout is shallow: fetch the reference commit, and export its tree (no worktree, so
# nothing is left in the checkout)
git -C "$REPO" fetch --no-tags --depth 1 origin "$REFERENCE_REF"
git -C "$REPO" archive FETCH_HEAD | tar -x -C "$WORKDIR/reference"

cd "$WORKDIR"
STATUS=0
section "Checks"
mkdir -p checks
(cd checks && python3 "$HERE/madmatrix_helicity_checks.py" --repo "$REPO" --reference-repo "$WORKDIR/reference" \
    --backends scalar "$BACKEND" 2>&1 | tee ../checks.log) || STATUS=1
section "Benchmark"
mkdir -p bench
(cd bench && python3 "$HERE/madmatrix_helicity_bench.py" --repo "$REPO" --reference-repo "$WORKDIR/reference" \
    --backends "$BACKEND" --nevt 65536 --me-nevt 16384 2>&1 | tee ../bench.log) || STATUS=1

section "Summary"
cat checks/summary.txt bench/summary.txt 2> /dev/null > summary.txt || true
cp bench/bench.md . 2> /dev/null || true
cat summary.txt
exit $STATUS
