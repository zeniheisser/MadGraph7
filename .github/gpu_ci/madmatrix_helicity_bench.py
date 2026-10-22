#!/usr/bin/env python3
"""Runtime of the branch-free helicities of madmatrix against a reference MadGraph7.

Run by madmatrix_helicity.yml (CPU backends) and by madmatrix_helicity_gpu.sh (job helicity
of gpu_runner_ci.yml, a GPU backend), in its run directory. --reference-repo is a checkout of
the commit before the branch-free helicities, whose wavefunction routines take the helicity
as an int and branch on it.

1. wavefunctions: wavefunction_bench.cc, built in a standalone output of each MadGraph7 and
   gauge (HelAmps_<model>.h of the unitary gauge, and of the FD gauge, helas_fd.h) with the
   flags of every backend, times every external wavefunction routine alone on --nevt random
   momenta, for every helicity: the events of a SIMD vector or GPU warp on one helicity
   (uniform, the good-helicity loop), or each on its own (mixed, a helicity given per
   event). The best of --rounds runs of each build, alternated, each the best of --ntrial
   timings of --nrep sweeps. The wavefunctions of the two builds must be the same (up to
   --rtol, relative to the largest component of the wavefunction), and those of the mixed
   sweep the same as the uniform ones.
2. matrix_elements: the time per event of the whole |M|^2 (check_sa.exe perf, the fastest of
   its iterations, best of --rounds runs) on the same momenta: the reference, the current one
   summed over the helicities, and the current one for one random helicity per event
   (UMAMI_IN_HELICITY_INDEX).

Writes bench.md (the tables) and summary.txt in the current directory. Only a failed build or
run, or different wavefunctions, fail (exit status 1); the timings are reported, not judged.
"""

import argparse
import array
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import madmatrix_checks as mc  # noqa: E402
from madmatrix_helicity_checks import prepare_repo, batch_of  # noqa: E402

TAGS = ('reference', 'current')

# (name, model, process, setup): one output per gauge for the wavefunction routines
WAVEFUNCTION_OUTPUTS = (
    ('unitary', 'sm', 'g g > t t~ z', ()),
    ('FD', 'sm', 'u u~ > w+ w-', ('set gauge FD',)),
)
# (name, model, process, energy) for the whole |M|^2
ME_PROCESSES = (
    ('ggttxz', 'sm', 'g g > t t~ z', 2000),
    ('uuxeeg', 'sm', 'u u~ > e+ e- g', 1000),
    ('ggttxgg', 'sm', 'g g > t t~ g g', 2000),
)

BENCH_MK = """# wavefunction_bench_<backend>.exe, written by madmatrix_helicity_bench.py
include makefile
wavefunction_bench_$(BACKEND).exe: $(BUILDDIR)/wavefunction_bench.o $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so
ifeq ($(GPUCC),)
\t$(CXX) -o $@ $< $(CXXLIBFLAGSRPATH) -L$(LIBDIR) -l$(MADMATRIX_COMMONLIB)
else
\t$(GPUCC) -o $@ $< $(GPULIBFLAGSRPATH) -L$(LIBDIR) -l$(MADMATRIX_COMMONLIB)
endif
"""


def run_output(cmd, cwd, log):
    """the stdout of cmd; RuntimeError (see log) if it fails"""
    with open(log, 'w') as out:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        out.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError('%s failed (exit status %d), see %s' % (' '.join(cmd), proc.returncode, log))
    return proc.stdout


def build_bench(proc_dir, backend, log):
    """wavefunction_bench_<backend>.exe in proc_dir, against its HelAmps_<model>.h"""
    src = os.path.join(proc_dir, '..', '..', 'src')
    (helamps,) = [f for f in os.listdir(src) if re.match(r'HelAmps_.*\.h$', f)]
    per_lane = 'fptype_amp_sv nhel' in open(os.path.join(src, helamps)).read()
    with open(os.path.join(proc_dir, 'wavefunction_bench_config.h'), 'w') as f:
        f.write('#define BENCH_PER_LANE_HELICITY %d\n#define BENCH_HELAMPS "%s"\n' % (per_lane, helamps))
    shutil.copy(os.path.join(HERE, 'wavefunction_bench.cc'), proc_dir)
    with open(os.path.join(proc_dir, 'bench.mk'), 'w') as f:
        f.write(BENCH_MK)
    exe = 'wavefunction_bench_%s.exe' % backend
    run_output(['make', '-f', 'bench.mk', 'BACKEND=' + backend, 'FPTYPE=d', exe], proc_dir, log)
    return os.path.abspath(os.path.join(proc_dir, exe)), per_lane


def parse_bench(text):
    """{(routine, mode): ns per call} from the RESULT lines (None: n/a)"""
    out = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0] == 'RESULT':
            out[fields[1], fields[2]] = None if fields[3] == 'n/a' else float(fields[3])
    return out


def compare_dumps(cur, ref, keys, nevt, nw6):
    """{routine: (max diff current vs reference, max diff mixed vs uniform current, the same for
    the reference)}: the largest |difference| of a component, relative to the largest component
    of the wavefunction; nan where there is nothing to compare"""
    block = 3 * nevt * nw6 * 2
    index = {key: i for i, key in enumerate(keys)}

    def wfs(dump, key):
        start = index[key] * block
        return dump[start:start + block]

    def diff(a, b):
        worst = 0.
        for start in range(0, len(b), nw6 * 2):
            wa, wb = a[start:start + nw6 * 2], b[start:start + nw6 * 2]
            if any(math.isnan(x) for x in wa) or any(math.isnan(x) for x in wb):
                if not (all(math.isnan(x) for x in wa) or all(math.isnan(x) for x in wb)):
                    return float('inf')
                return float('nan')  # not available (an n/a sweep)
            scale = max(abs(x) for x in wb)
            d = max(abs(x - y) for x, y in zip(wa, wb))
            worst = max(worst, d / scale if scale > 0 else (0. if d == 0 else float('inf')))
        return worst

    out = {}
    for routine in sorted({k[0] for k in keys}, key=lambda r: index[r, 'uniform']):
        uniform = (routine, 'uniform')
        mixed = (routine, 'mixed')
        out[routine] = (diff(wfs(cur, uniform), wfs(ref, uniform)),
                        diff(wfs(cur, mixed), wfs(cur, uniform)) if mixed in index else float('nan'),
                        diff(wfs(ref, mixed), wfs(ref, uniform)) if mixed in index else float('nan'))
    return out


def fmt(value, spec='%.2f'):
    return 'n/a' if value is None or (isinstance(value, float) and math.isnan(value)) else spec % value


def bench_wavefunctions(args, report, problems):
    mc.section('external wavefunction routines alone')
    for gauge, model, process, setup in WAVEFUNCTION_OUTPUTS:
        dirs = {}
        for tag in TAGS:
            repo = args.repo if tag == 'current' else args.reference_repo
            name = 'bench_wf_%s_%s' % (gauge, tag)
            # only the headers of src are needed: no backend built by output, bench.mk builds it all
            (dirs[tag],) = mc.output(repo, name, model, process, ())
        for backend in args.backends:
            exes, per_lane, results, dumps = {}, {}, {tag: {} for tag in TAGS}, {}
            for tag in TAGS:
                exes[tag], per_lane[tag] = build_bench(dirs[tag], backend, 'bench_wf_%s_%s_%s_make.log'
                                                       % (gauge, tag, backend))
            for iround in range(args.rounds):
                for tag in (TAGS if iround % 2 == 0 else TAGS[::-1]):
                    dump = os.path.abspath('bench_wf_%s_%s_%s.bin' % (gauge, tag, backend))
                    cmd = [exes[tag], str(args.nevt), str(args.nrep), str(args.ntrial)] + ([dump] if iround == 0 else [])
                    text = run_output(cmd, os.path.dirname(exes[tag]),
                                      'bench_wf_%s_%s_%s_%d.log' % (gauge, tag, backend, iround))
                    if iround == 0:
                        header = text.splitlines()[0]
                        nevt = int(re.search(r'nevt=(\d+)', header).group(1))
                        ndump = int(re.search(r'ndump=(\d+)', header).group(1))
                        nw6 = int(re.search(r'nw6=(\d+)', header).group(1))
                        dumps[tag] = array.array('d')
                        with open(dump, 'rb') as f:
                            dumps[tag].frombytes(f.read())
                    for key, ns in parse_bench(text).items():
                        old = results[tag].get(key)
                        results[tag][key] = ns if old is None or (ns is not None and ns < old) else old
            keys = list(parse_bench(text).keys())
            diffs = compare_dumps(dumps['current'], dumps['reference'], keys, ndump, nw6)
            report.append('\n### Wavefunction routines, %s gauge, %s\n' % (gauge, backend))
            report.append('%d random momenta, ns per call and event (best of %d x %d), before: helicity an int '
                          '(per-lane helicity: %s), after: per lane (%s). Max difference of the wavefunctions, '
                          'relative to their largest component.\n'
                          % (nevt, args.rounds, args.ntrial, bool(per_lane['reference']), bool(per_lane['current'])))
            report.append('| routine | before uniform | after uniform | ratio | before mixed | after mixed | ratio '
                          '| after vs before | mixed vs uniform |')
            report.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
            for routine, (d_ref, d_mix, d_mix_ref) in diffs.items():
                row = [routine]
                for mode in ('uniform', 'mixed'):
                    b, a = results['reference'].get((routine, mode)), results['current'].get((routine, mode))
                    row += [fmt(b), fmt(a), fmt(a / b if a is not None and b else None)]
                row += [fmt(d_ref, '%.1e'), fmt(d_mix, '%.1e')]
                report.append('| ' + ' | '.join(row) + ' |')
                where = '%s gauge %s %s' % (gauge, backend, routine)
                if not d_ref <= args.rtol:
                    problems.append('%s: the wavefunctions differ from the reference (%.1e)' % (where, d_ref))
                if not (math.isnan(d_mix) or d_mix <= args.rtol):
                    problems.append('%s: a helicity per event gives other wavefunctions (%.1e)' % (where, d_mix))
                if not (math.isnan(d_mix_ref) or d_mix_ref <= args.rtol):
                    problems.append('%s: reference, a helicity per event gives other wavefunctions' % where)
            print('\n'.join(report[-len(diffs) - 4:]), flush=True)


def check_sa_time(proc_dir, backend, args, log):
    """the fastest iteration of check_sa_<backend>.exe perf <args>, in seconds"""
    mc.check_sa(proc_dir, backend, args, log)
    text = open(log).read()
    match = re.search(r'\[Min,Max\]TimeInMatrixElems\s*=\s*\[\s*(\S+)\s*,', text)
    if not match:
        raise RuntimeError('no [Min,Max]TimeInMatrixElems in %s' % log)
    return float(match.group(1))


def bench_matrix_elements(args, report, problems):
    mc.section('the whole |M|^2')
    report.append('\n### Whole |M|^2 (check_sa.exe perf)\n')
    report.append('%d events an iteration, %d iterations, microseconds per event of the fastest iteration '
                  '(best of %d runs); the same momenta for all.\n' % (args.me_nevt, args.me_niter, args.rounds))
    report.append('| process | backend | before | after | ratio | after, one random helicity per event | ratio |')
    report.append('|---|---|---:|---:|---:|---:|---:|')
    rng = random.Random(23)
    for name, model, process, energy in ME_PROCESSES:
        dirs = {}
        for tag in TAGS:
            repo = args.repo if tag == 'current' else args.reference_repo
            (dirs[tag],) = mc.output(repo, 'bench_me_%s_%s' % (name, tag), model, process, args.backends)
        batch = batch_of(args.me_nevt)[:2] + [args.me_niter]
        momenta = mc.dump('bench_me_' + name, 'momenta')
        mc.check_sa(dirs['current'], args.backends[0], ['--energy', energy, '--dump-momenta', momenta] + batch,
                    'bench_me_%s_rambo.log' % name)
        ncomb = int(re.search(r'constexpr int ncomb = (\d+);',
                              open(os.path.join(dirs['current'], 'ProcessData.h')).read()).group(1))
        hels = mc.write('bench_me_%s_hel.bin' % name, 'i',
                        [rng.randrange(ncomb) for _ in range(args.me_nevt * args.me_niter)])
        for backend in args.backends:
            best = {}
            runs = (('before', 'reference', []), ('after', 'current', []),
                    ('one', 'current', ['--hel-index', hels]))
            for iround in range(args.rounds):
                for label, tag, extra in (runs if iround % 2 == 0 else runs[::-1]):
                    seconds = check_sa_time(dirs[tag], backend, ['--momenta', momenta] + extra + batch,
                                            'bench_me_%s_%s_%s_%d.log' % (name, label, backend, iround))
                    best[label] = min(best.get(label, seconds), seconds)
            us = {label: best[label] / args.me_nevt * 1e6 for label in best}
            report.append('| %s | %s | %.3f | %.3f | %.2f | %.3f | %.2f |'
                          % (process, backend, us['before'], us['after'], us['after'] / us['before'],
                             us['one'], us['one'] / us['before']))
            print(report[-1], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', required=True, help='the MadGraph7 under test')
    parser.add_argument('--reference-repo', required=True, help='the reference MadGraph7')
    parser.add_argument('--backends', nargs='+', required=True)
    parser.add_argument('--nevt', type=int, default=4096, help='random momenta of the wavefunction benchmark')
    parser.add_argument('--nrep', type=int, default=200, help='sweeps over all of them a timing')
    parser.add_argument('--ntrial', type=int, default=5, help='timings a run (the best is kept)')
    parser.add_argument('--rounds', type=int, default=3, help='runs of each build, alternated')
    parser.add_argument('--me-nevt', type=int, default=4096, help='events an iteration of check_sa.exe')
    parser.add_argument('--me-niter', type=int, default=5, help='iterations of check_sa.exe')
    parser.add_argument('--rtol', type=float, default=1e-12)
    parser.add_argument('--only', choices=('wavefunctions', 'matrix_elements'), action='append')
    args = parser.parse_args()
    start = time.time()
    prepare_repo(args.repo)
    prepare_repo(args.reference_repo)
    report = ['## Branch-free helicities: runtime against the reference MadGraph7',
              '', 'node %s, %s' % (os.uname().nodename, os.environ.get('GPU_NAME') or cpu_name())]
    failures = []
    for part in args.only or ('wavefunctions', 'matrix_elements'):
        problems = []
        try:
            globals()['bench_' + part](args, report, problems)
        except Exception as error:
            problems.append('error: %s' % error)
        mc.check(part, not problems, '; '.join(problems[:4]) or 'done')
        if problems:
            failures.append(part)
            report.append('\n**%s FAILED**: %s\n' % (part, '; '.join(problems[:4])))
    with open('bench.md', 'w') as f:
        f.write('\n'.join(report) + '\n')
    with open('summary.txt', 'w') as f:
        for key, (ok, detail) in mc.results.items():
            f.write('%s=%s (%s)\n' % (key, 'ok' if ok else 'FAILED', detail))
        f.write('backends=%s\nwalltime=%ds\n' % (' '.join(args.backends), time.time() - start))
    return 1 if failures else 0


def cpu_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return 'unknown CPU'


if __name__ == '__main__':
    sys.exit(main())
