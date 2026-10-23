#!/usr/bin/env python3
"""Branch-free helicities and single-helicity evaluation of madmatrix, with check_sa.exe.

Run by madmatrix_helicity.yml (CPU backends) and by madmatrix_helicity_gpu.sh (job helicity of
gpu_runner_ci.yml, a GPU backend), in its run directory. The standalone outputs of a set of
processes that, together, use every external wavefunction routine for which the helicity
matters (massive and massless ixxxxx/oxxxxx, massive and massless vxxxxx with a
longitudinal polarisation, Majorana fermions, the FD gauge of helas_fd.h, and sxxxxx), are
evaluated by the check_sa.exe of every backend asked for, on the same events: the momenta
and random numbers go in as raw binary files, and the |M|^2, helicities and colours come out
the same way (see madmatrix_checks.py).

1. reference: the |M|^2, the chosen helicity and the chosen colour of every event are those
   of a reference MadGraph7 (--reference-repo, a checkout of the commit before the
   branch-free helicities), backend by backend, up to the numerical precision (--rtol): the
   branch-free arithmetic must not change any result.
2. polarised_sum: for each of the ncomb helicities, given to every event
   (UMAMI_IN_HELICITY_INDEX, --hel-index), the |M|^2 of that helicity; their sum must be the
   unpolarised (helicity-summed) |M|^2, event by event, which also checks that the colour sum
   is done for the one helicity. With a random helicity per event (the lanes of a SIMD vector,
   or the threads of a GPU warp, on different helicities), every event must get the |M|^2 of
   its helicity and return it as the chosen helicity. Every backend gives the |M|^2 of the
   scalar backend, helicity by helicity (up to 1e-9 of the unpolarised |M|^2: a helicity that
   vanishes analytically is rounding noise, which differs between backends).

Writes summary.txt (key=value lines) in the current directory; exit status 1 if a check
fails.
"""

import argparse
import math
import os
import random
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import madmatrix_checks as mc  # noqa: E402 (output, check_sa, read, write, dump, check)

CHECKS = ('reference', 'polarised_sum')

# (name, model, process, setup commands, energy): every wavefunction routine with a helicity
PROCESSES = (
    # massless and massive vectors (a longitudinal Z), massive fermions
    ('ggttxz', 'sm', 'g g > t t~ z', (), 2000),
    # massless fermions in and out, a massless vector
    ('uuxeeg', 'sm', 'u u~ > e+ e- g', (), 1000),
    # a scalar, massive fermions; two subprocesses, the light quarks grouped by flavour
    ('pp_ttxh', 'sm', 'p p > t t~ h', (), 2000),
    # the FD gauge (helas_fd.h): the fifth component of the vector wavefunctions
    ('uuxwwfd', 'sm', 'u u~ > w+ w-', ('set gauge FD',), 1000),
    # Majorana fermions
    ('uuxn1n2', 'MSSM_SLHA2', 'u u~ > n1 n2', (), 2000),
)


def prepare_repo(repo):
    """the default configuration of a MadGraph7 checkout, as .github/actions/checkout_mg5"""
    conf = os.path.join(repo, 'input', 'mg7_configuration.txt')
    if not os.path.exists(conf):
        shutil.copy(os.path.join(repo, 'input', '.mg7_configuration_default.txt'), conf)
        with open(conf, 'a') as f:
            f.write('auto_update = 0\nautomatic_html_opening = False\nnotification_center = False\n'
                    'nb_core = %s\n' % os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    make_opts = os.path.join(repo, 'Template', 'LO', 'Source', 'make_opts')
    if not os.path.exists(make_opts):
        shutil.copy(os.path.join(repo, 'Template', 'LO', 'Source', '.make_opts'), make_opts)


def batch_of(nevt):
    """check_sa.exe perf arguments for nevt events in one iteration (GPU: 256 threads a block)"""
    return [max(1, nevt // 256), min(nevt, 256), 1]


def ncomb_of(proc_dir):
    data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
    return int(re.search(r'constexpr int ncomb = (\d+);', data).group(1))


def rel_diff(a, b):
    """|a - b| / |b|, 0 if both vanish, inf if only b does; nan stays nan"""
    if a == b:
        return 0.
    return abs(a - b) / abs(b) if b != 0 else float('inf')


def worst(rels):
    """the largest finite relative difference, and the events beyond tolerance (nan included:
    it fails every comparison)"""
    return max((r for r in rels if not math.isnan(r)), default=0.)


class Outputs:
    """the standalone output of every process, built for every backend, with the current
    MadGraph7 and (for check 1) the reference one; the same momenta and random numbers for
    both, per subprocess"""

    def __init__(self, repo, reference_repo, backends, nevt, seed=7):
        self.repo, self.reference_repo, self.backends, self.nevt = repo, reference_repo, backends, nevt
        self.rng = random.Random(seed)
        self.dirs = {}    # (tag, name) -> subprocess directories
        self.inputs = {}  # (name, subprocess) -> (momenta, rnd-hel, rnd-col) files

    def get(self, tag, name):
        if (tag, name) not in self.dirs:
            (_name, model, process, setup, _energy) = [p for p in PROCESSES if p[0] == name][0]
            repo = self.repo if tag == 'current' else self.reference_repo
            mc.section('output %s (%s) with the %s MadGraph7, backends %s'
                       % (process, model, tag, ' '.join(self.backends)))
            self.dirs[tag, name] = mc.output(repo, '%s_%s' % (tag, name), model, process,
                                             self.backends, setup)
        return self.dirs[tag, name]

    def events(self, name, proc_dir):
        """the momenta (RAMBO points of the current scalar check_sa.exe) and the random
        numbers of the helicity and colour choices of a subprocess"""
        sub = os.path.basename(proc_dir)
        if (name, sub) not in self.inputs:
            energy = [p for p in PROCESSES if p[0] == name][0][4]
            prefix = '%s_%s' % (name, sub)
            momenta = mc.dump(prefix, 'momenta')
            mc.check_sa(proc_dir, self.backends[0], ['--energy', energy, '--dump-momenta', momenta]
                        + batch_of(self.nevt), prefix + '_rambo.log')
            rnd_hel = mc.write(prefix + '_rndhel.bin', 'd', [self.rng.random() for _ in range(self.nevt)])
            rnd_col = mc.write(prefix + '_rndcol.bin', 'd', [self.rng.random() for _ in range(self.nevt)])
            self.inputs[name, sub] = (momenta, rnd_hel, rnd_col)
        return self.inputs[name, sub]

    def run(self, tag, name, proc_dir, backend, extra=(), label=''):
        """|M|^2, helicities and colours of check_sa_<backend>.exe on the events of proc_dir"""
        momenta, rnd_hel, rnd_col = self.events(name, proc_dir)
        prefix = '%s_%s_%s_%s%s' % (tag, name, os.path.basename(proc_dir), backend, label)
        mc.check_sa(proc_dir, backend, ['--momenta', momenta, '--rnd-hel', rnd_hel, '--rnd-col', rnd_col,
                                        '--dump-me', mc.dump(prefix, 'me'), '--dump-hel', mc.dump(prefix, 'hel'),
                                        '--dump-col', mc.dump(prefix, 'col')] + list(extra) + batch_of(self.nevt),
                    prefix + '.log')
        return (mc.read(mc.dump(prefix, 'me'), 'd'), mc.read(mc.dump(prefix, 'hel'), 'i'),
                mc.read(mc.dump(prefix, 'col'), 'i'))


# ---------------------------------------------------------------------------
# 1. the same results as the reference MadGraph7
# ---------------------------------------------------------------------------

def check_reference(outputs, rtol):
    mc.section('the same |M|^2, helicities and colours as the reference MadGraph7')
    if not outputs.reference_repo:
        mc.check('reference', False, 'no --reference-repo')
        return
    problems, details = [], []
    for name, _model, process, _setup, _energy in PROCESSES:
        try:
            current, reference = outputs.get('current', name), outputs.get('reference', name)
        except Exception as error:
            problems.append('%s: %s' % (process, error))
            continue
        if [os.path.basename(d) for d in current] != [os.path.basename(d) for d in reference]:
            problems.append('%s: subprocesses %s, reference %s' % (process, current, reference))
            continue
        for cur_dir, ref_dir in zip(current, reference):
            sub = os.path.basename(cur_dir)
            for backend in outputs.backends:
                me, hel, col = outputs.run('current', name, cur_dir, backend)
                ref_me, ref_hel, ref_col = outputs.run('reference', name, ref_dir, backend)
                rels = [rel_diff(a, b) for a, b in zip(me, ref_me)]
                bad = [i for i, r in enumerate(rels) if not r <= rtol]
                identical = sum(1 for a, b in zip(me, ref_me) if a == b)
                bad_hel = sum(1 for a, b in zip(hel, ref_hel) if a != b)
                bad_col = sum(1 for a, b in zip(col, ref_col) if a != b)
                where = '%s %s %s' % (process, sub, backend)
                if len(me) != outputs.nevt or len(ref_me) != outputs.nevt:
                    problems.append('%s: %d and %d |M|^2 for %d events' % (where, len(me), len(ref_me), outputs.nevt))
                if bad:
                    i = bad[0]
                    problems.append('%s: %d of %d |M|^2 differ, first event %d: %.16e vs %.16e (rel %.1e)'
                                    % (where, len(bad), len(me), i, me[i], ref_me[i], rels[i]))
                if bad_hel or bad_col:
                    problems.append('%s: %d chosen helicities and %d chosen colours differ'
                                    % (where, bad_hel, bad_col))
                if not any(v > 0 for v in me):
                    problems.append('%s: no |M|^2 > 0' % where)
                details.append('%s: max rel diff %.1e, %d/%d bit-identical' % (where, worst(rels), identical, len(me)))
                print('  ' + details[-1], flush=True)
    mc.check('reference', not problems, '; '.join(problems[:4]) or '; '.join(details))


# ---------------------------------------------------------------------------
# 2. the sum of the polarised |M|^2 is the unpolarised one
# ---------------------------------------------------------------------------

def check_polarised_sum(outputs, rtol):
    mc.section('the polarised |M|^2 add up to the unpolarised one')
    problems, details = [], []
    rng = random.Random(17)
    for name, _model, process, _setup, _energy in PROCESSES:
        try:
            proc_dirs = outputs.get('current', name)
        except Exception as error:
            problems.append('%s: %s' % (process, error))
            continue
        for proc_dir in proc_dirs:
            sub = os.path.basename(proc_dir)
            ncomb = ncomb_of(proc_dir)
            nevt = outputs.nevt
            mixed = [rng.randrange(ncomb) for _ in range(nevt)]
            mixed_file = mc.write('%s_%s_mixed_in.bin' % (name, sub), 'i', mixed)
            perhel, unpolarised = {}, {}
            for backend in outputs.backends:
                where = '%s %s %s' % (process, sub, backend)
                summed, _hel, _col = outputs.run('current', name, proc_dir, backend)
                unpolarised[backend] = summed
                perhel[backend] = []
                for ihel in range(ncomb):
                    hel_file = mc.write('%s_%s_h%d_in.bin' % (name, sub, ihel), 'i', [ihel] * nevt)
                    me, hel, _col = outputs.run('current', name, proc_dir, backend,
                                                ['--hel-index', hel_file], '_h%d' % ihel)
                    perhel[backend].append(me)
                    if set(hel) != {ihel}:
                        problems.append('%s: helicity %d returned as %s' % (where, ihel, sorted(set(hel))[:4]))
                    if not all(v >= 0 for v in me):
                        problems.append('%s: helicity %d has a |M|^2 < 0 (or nan)' % (where, ihel))
                total = [math.fsum(perhel[backend][ihel][i] for ihel in range(ncomb)) for i in range(nevt)]
                rels = [rel_diff(a, b) for a, b in zip(total, summed)]
                bad = [i for i, r in enumerate(rels) if not r <= rtol]
                if bad or len(summed) != nevt:
                    i = bad[0] if bad else 0
                    problems.append('%s: the sum of the %d polarised |M|^2 differs from the unpolarised one in '
                                    '%d of %d events (event %d: %.16e vs %.16e)'
                                    % (where, ncomb, len(bad), nevt, i, total[i], summed[i]))
                if not all(v > 0 for v in summed):
                    problems.append('%s: unpolarised |M|^2 not all > 0' % where)
                nzero = sum(1 for ihel in range(ncomb) if not any(perhel[backend][ihel]))
                # a random helicity per event, on the same events
                me, hel, _col = outputs.run('current', name, proc_dir, backend, ['--hel-index', mixed_file], '_mixed')
                expected = [perhel[backend][ihel][i] for i, ihel in enumerate(mixed)]
                bad_me = sum(1 for a, b, s in zip(me, expected, summed) if not abs(a - b) <= rtol * s)
                bad_hel = sum(1 for a, b in zip(hel, mixed) if a != b)
                if bad_me or bad_hel or len(me) != nevt:
                    problems.append('%s: a random helicity per event: %d |M|^2 differ from that helicity\'s, '
                                    '%d helicities not returned' % (where, bad_me, bad_hel))
                details.append('%s: %d helicities (%d vanish), max rel diff of the sum %.1e'
                               % (where, ncomb, nzero, worst(rels)))
                print('  ' + details[-1], flush=True)
            # every backend against the scalar one (the first), helicity by helicity, relative to
            # the unpolarised |M|^2 of the event: a helicity that vanishes is rounding noise
            # (~1e-40), which differs between backends
            first = outputs.backends[0]
            for backend in outputs.backends[1:]:
                rels = [abs(a - b) / s for ihel in range(ncomb)
                        for a, b, s in zip(perhel[backend][ihel], perhel[first][ihel], unpolarised[first])]
                bad = sum(1 for r in rels if not r <= 1e-9)
                if bad:
                    problems.append('%s %s: %d (helicity, event) |M|^2 differ between %s and %s'
                                    % (process, sub, bad, backend, first))
                details.append('%s %s %s vs %s: max diff of a polarised |M|^2 %.1e (of the unpolarised one)'
                               % (process, sub, backend, first, worst(rels)))
                print('  ' + details[-1], flush=True)
    mc.check('polarised_sum', not problems, '; '.join(problems[:4]) or '; '.join(details))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', required=True, help='the MadGraph7 under test')
    parser.add_argument('--reference-repo', help='the reference MadGraph7 (check reference)')
    parser.add_argument('--backends', nargs='+', required=True,
                        help='backends, the first one (scalar) the reference of the others')
    parser.add_argument('--nevt', type=int, default=1024, help='events per subprocess (a multiple of 256)')
    parser.add_argument('--rtol', type=float, default=1e-12,
                        help='relative tolerance of the |M|^2 comparisons of one backend')
    parser.add_argument('--only', choices=CHECKS, action='append', help='run this check only (repeatable)')
    args = parser.parse_args()
    start = time.time()
    prepare_repo(args.repo)
    if args.reference_repo:
        prepare_repo(args.reference_repo)
    outputs = Outputs(args.repo, args.reference_repo, args.backends, args.nevt)
    for name in args.only or CHECKS:
        try:
            globals()['check_' + name](outputs, args.rtol)
        except Exception as error:
            mc.check(name, False, 'error: %s' % error)
    summary = {key: ('ok' if ok else 'FAILED') + ' (%s)' % detail for key, (ok, detail) in mc.results.items()}
    failures = [key for key, (ok, _detail) in mc.results.items() if not ok]
    summary['backends'] = ' '.join(args.backends)
    summary['node'] = os.uname().nodename
    summary['walltime'] = '%ds' % (time.time() - start)
    with open('summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write('%s=%s\n' % (key, value))
    print('\n%d check(s) failed: %s' % (len(failures), ', '.join(failures)) if failures
          else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
