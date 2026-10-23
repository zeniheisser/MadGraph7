import argparse
import glob
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Locate the madspace installation bundled alongside MadGraph, same layout
# launch.py resolves. Unlike launch.py, this module does not trigger the
# installer itself: by the time reweighting is needed, generate_events has
# already ensured the install exists.
import madgraph as _mg_pkg
_MG_ROOT = Path(_mg_pkg.__file__).parents[1]
_INSTALL_DIR = _MG_ROOT / "madspace" / "install"
if (_INSTALL_DIR / "madspace").is_dir() and str(_INSTALL_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALL_DIR))

import madspace as ms
import madspace.madtrex as mtx
from madgraph.various import misc
from madgraph.various.banner import RunCardMG7

from . import param_reweight_card as param_rwgt_card
from . import pdf_reweight_card as pdf_rwgt_card
from . import generic_reweight_card as generic_rwgt_card

logger = logging.getLogger("madgraph7")


def find_latest_run_dir(events_dir: str = "Events") -> str:
    """Returns the most recently modified Events/<run_name>_NN run directory."""
    run_dirs = glob.glob(os.path.join(events_dir, "*_[0-9][0-9]"))
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {events_dir}")
    return max(run_dirs, key=os.path.getmtime)


def find_run_events(run_dir: str) -> str:
    """Returns run_dir's generated events file, preferring LHEF XML over the
    internal binary format (see launch.py's generate_events, which writes
    events.lhe.gz for output_format="lhe" and events.npy for "lhe_npy")."""
    for name in ("events.lhe", "events.lhe.gz", "events.npy"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no events.lhe(.gz) or events.npy found in {run_dir}")


def prepare_events(events_path: str) -> str:
    """Returns a path MadtRex can read events_path from: the MadtRex readers
    don't handle gzip, so a gzipped LHE file (launch.py gzips events.lhe by
    default) is decompressed once into a sibling file without the .gz suffix,
    reused on later calls. Any other path is returned unchanged."""
    if not events_path.endswith(".gz"):
        return events_path
    plain_path = events_path[: -len(".gz")]
    if not os.path.isfile(plain_path) or os.path.getmtime(plain_path) < os.path.getmtime(events_path):
        logger.info("MadtRex: decompressing %s -> %s", events_path, plain_path)
        with gzip.open(events_path, "rb") as src, open(plain_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return plain_path


def read_init_node(events_path: str) -> "ms.LHEMeta | None":
    """Reads the <init> block launch.py's MadgraphProcess.write_lhe_header()
    writes as header.lhe next to events_path (or, for runs made before that
    existed, the init.lhe the old madevent.py wrote), if present, into an
    LHEMeta that
    Driver.load_events() can use to seed the reweighted sample's baseline
    cross section and error. Needed for events.npy input (compact_npy and
    lhe_npy), which -- unlike events.lhe -- carries no <init> block of its
    own; without this, MadtRex reweighting has no baseline cross section to
    normalise or propagate errors against (see reweightor::calc_xErrs() /
    calc_norm() in Rex/src/teaRex.cc). Returns None if neither file is
    found, in which case load_events() falls back to its own empty-LHEMeta
    default."""
    run_dir = os.path.dirname(events_path)
    for name in ("header.lhe", "init.lhe"):
        init_path = os.path.join(run_dir, name)
        if os.path.isfile(init_path):
            break
    else:
        return None
    with open(init_path) as f:
        match = re.search(r"<init>\s*\n(.*?)</init>", f.read(), re.S)
    if not match:
        return None
    lines = [line.split() for line in match.group(1).strip().splitlines()]
    header, proc_lines = lines[0], lines[1:]

    def process(fields: list[str]) -> ms.LHEProcess:
        proc = ms.LHEProcess(
            cross_section=float(fields[0]), cross_section_error=float(fields[1]),
            process_id=int(fields[3]),
        )
        proc.max_weight = float(fields[2])  # constructor kwarg is misspelled in the binding
        return proc

    return ms.LHEMeta(
        beam1_pdg_id=int(header[0]), beam2_pdg_id=int(header[1]),
        beam1_energy=float(header[2]), beam2_energy=float(header[3]),
        beam1_pdf_authors=int(header[4]), beam2_pdf_authors=int(header[5]),
        beam1_pdf_id=int(header[6]), beam2_pdf_id=int(header[7]),
        weight_mode=int(header[8]),
        processes=[process(fields) for fields in proc_lines],
    )


def _pdf_data_path() -> str | None:
    """Resolves the LHAPDF data directory: $LHAPDF_DATA_PATH if set, else
    lhapdf.paths()[0] (the configured lhapdf's own data dir, eg lhapdf6 in
    HEPTools) if the lhapdf Python module is importable, else None -- same
    convention as launch.py's own PDF_PATH."""
    if "LHAPDF_DATA_PATH" in os.environ:
        return os.environ["LHAPDF_DATA_PATH"]
    try:
        import lhapdf
        lhapdf.setVerbosity(0)
        return lhapdf.paths()[0]
    except ImportError:
        return None


def resolve_pdf_set_files(pdf_set_name: str) -> tuple[str, str]:
    """Resolves pdf_set_name (an LHAPDF set -- the only kind PdfDriver
    currently supports) to its (grid_file, alpha_s_grid_file) paths, per the
    same $LHAPDF_DATA_PATH/lhapdf.paths()[0] convention launch.py's own
    init_beam()/ensure_pdf_set() use for the run_card's own [beam] pdf.
    Unlike ensure_pdf_set(), this never attempts to download a missing set:
    standalone reweighting has no run_card.toml/MADGRAPH_LHAPDF_CONFIG to
    drive a download from, so it only resolves paths, raising a clear error
    if pdf_set_name isn't already present on disk."""
    data_path = _pdf_data_path()
    if data_path is None:
        raise RuntimeError(
            f"cannot resolve PDF set {pdf_set_name!r}: no LHAPDF_DATA_PATH is set and the "
            f"lhapdf Python module is not importable"
        )
    set_dir = os.path.join(data_path, pdf_set_name)
    if not os.path.isdir(set_dir):
        raise RuntimeError(
            f"PDF set {pdf_set_name!r} not found under {data_path!r}; install it (eg via "
            f"MadGraph's `install` command, or lhapdf's own `lhapdf install`) before "
            f"reweighting against it"
        )
    return (
        os.path.join(set_dir, f"{pdf_set_name}_0000.dat"),
        os.path.join(set_dir, f"{pdf_set_name}.info"),
    )


def default_output_path(events_path: str) -> str:
    root, ext = os.path.splitext(events_path)
    if ext == ".npy":
        # write_weights() targets save_weights_binary() for binary input,
        # which treats its path argument as a prefix, writing one
        # "<prefix>_<weight_id>.npy" file per recorded weight -- so the
        # default must not carry the .npy extension itself.
        return f"{root}_reweighted"
    return f"{root}_reweighted{ext}"


def _read_devices(process_directory: str) -> list[str]:
    """The matrix-element build backends (the BACKEND= / {device} values) of
    process_directory's run_card.toml: [run] device mapped through cpu_mode
    as launch.py's backend_of() does, with cpu_mode 'auto' resolved to the
    concrete SIMD backend so it matches the library names on disk."""
    run_card = RunCardMG7(os.path.join(process_directory, "Cards", "run_card.toml"))
    device_names = run_card["run"]["device"]
    if not isinstance(device_names, list):
        device_names = [device_names]
    cpu_mode = run_card["run"]["cpu_mode"]
    backends = []
    for name in device_names:
        device_type = name.split(":", 1)[0]
        backends.append(cpu_mode if device_type == "cpu" else device_type)
    if "auto" in backends:
        # launch.py is only imported when needed: importing it bootstraps
        # the madspace install, which this module otherwise leaves alone
        from .launch import resolve_auto_backend
        with open(os.path.join(process_directory, "SubProcesses", "subprocesses.json")) as f:
            first_proc_path = json.load(f)[0]["path"]
        if not os.path.isabs(first_proc_path):
            first_proc_path = os.path.join(process_directory, first_proc_path)
        auto_backend = resolve_auto_backend(first_proc_path)
        backends = [auto_backend if b == "auto" else b for b in backends]
    return list(dict.fromkeys(backends))


def _read_default_pdf_set(process_directory: str) -> str | None:
    """Best-effort read of [beam] pdf1 from process_directory/Cards/
    run_card.toml -- the PDF set the run itself actually used to generate
    its events, per the same run_card field launch.py's init_beam() reads
    (PdfDriver needs the same set on both beams; a run with pdf1 != pdf2
    returns None, so the card has to name its baseline explicitly). This is run_pdf_reweighting()'s hook for inferring a
    PDF reweight card's baseline PDF set (see pdf_reweight_card.py's module
    docstring) when the card's own preamble doesn't set one explicitly:
    since the run_card is the actual generation record, its [beam] pdf1 is a
    reliable stand-in for "whatever PDF set this sample's own weight
    already corresponds to" -- see _read_run_fptype() just below for the
    equivalent hook for FPTYPE.

    Returns None, not an error, whenever the run_card doesn't exist or
    doesn't carry a usable pdf entry (eg process_directory is a bare/
    relocated sample with no Cards/ of its own, or a leptonic run) --
    callers fall back to requiring an explicit
    PDF_SET in that case, same as before this hook existed."""
    run_card_path = os.path.join(process_directory, "Cards", "run_card.toml")
    if not os.path.isfile(run_card_path):
        return None
    try:
        beam = RunCardMG7(run_card_path)["beam"]
        if beam["leptonic"] or beam["pdf1"] != beam["pdf2"]:
            return None
        pdf_set = beam["pdf1"]
    except (KeyError, TypeError):
        return None
    return pdf_set or None


# [run] precision -> FPTYPE, as launch.py's fptype_of()
_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def _read_run_fptype(process_directory: str) -> str | None:
    """Best-effort read of the FPTYPE from process_directory/Cards/
    run_card.toml's [run] precision -- the fptype this sample was ACTUALLY
    generated/compiled at (mapped as launch.py's fptype_of() does; see
    MadgraphProcess.compile_matrix_elements in launch.py). Used by
    run_parameter_reweighting() for two, related purposes:

    - as the default *target* fptype (param_rwgt_card.group_by_fptype()'s
      default_fptype) when a reweight card sets no "set FPTYPE ..." at all,
      so a plain, non-FPTYPE-changing reweight stays at whatever precision
      the sample already is, rather than drifting to a hardcoded default;
    - as the *normaliser* fptype (Driver.set_normaliser_fptype_priority()),
      always -- a launch must be normalised against the sample's real,
      original configuration, never against whichever fptype it's being
      reweighted TO. This is the "future hook" param_reweight_card.py's own
      module docstring flagged as missing; the run_card is the actual
      generation record, so this is a real fact being looked up, not a
      guessed default (same reasoning as _read_default_pdf_set() above).

    Returns None, not an error, on any bare/relocated/predates-the-fptype-
    field process_directory -- callers fall back to
    param_rwgt_card.DEFAULT_FPTYPE in that case, same as group_by_fptype()'s
    own pre-existing fallback."""
    run_card_path = os.path.join(process_directory, "Cards", "run_card.toml")
    if not os.path.isfile(run_card_path):
        return None
    try:
        precision = RunCardMG7(run_card_path)["run"]["precision"]
    except (KeyError, TypeError):
        return None
    if not precision:
        return None
    precision = str(precision).lower()
    return _PRECISION_2_FPTYPE.get(precision, precision)


def _compile_subprocesses(directory: str, devices: list[str], fptype: str) -> None:
    """Compiles whichever subprocesses under
    directory/SubProcesses/subprocesses.json aren't already compiled for
    fptype, for each of devices (BACKEND=<device> FPTYPE=<fptype>) --
    mirrors the existence-checked per-device compile in launch.py's
    MadgraphProcess.compile_matrix_elements, generalized to whichever fptype a reweight
    card's "set FPTYPE ..." (see reweight_card.py's group_by_fptype()) or a
    caller actually asked for. Safe (a fast no-op) to call for a
    device/fptype combination that's already built, so callers needn't
    track that themselves."""
    with open(os.path.join(directory, "SubProcesses", "subprocesses.json")) as f:
        subprocesses = json.load(f)
    for entry in subprocesses:
        for device in devices:
            expected_path = os.path.join(directory, entry["me_path"].format(device=device, fptype=fptype))
            if os.path.isfile(expected_path):
                continue
            logger.info(
                "MadtRex: compiling %s for device '%s', fptype '%s'",
                entry["path"], device, fptype,
            )
            # madmatrix.mk's own USEBUILDDIR=1 build-directory tag
            # (BACKEND+FPTYPE+HELINL+HRDCOD) refuses to build a *different*
            # tag while another tag's now-stale intermediate .o files are
            # still sitting in build.<device>/ from an earlier compile (eg
            # the original generation's own FPTYPE, or a previous reweight
            # card's) -- "Cannot build for tag=... as old builds exist for
            # other tags ...: Please run 'make clean' first". Harmless to
            # always do so before compiling here: this branch only runs
            # when expected_path doesn't exist yet, ie a real compile is
            # needed regardless, and the *already-built* .so for whatever
            # other tag was there previously lives in lib/, untouched by
            # 'clean' removing only build.<device>/'s own intermediates.
            # 'clean' here (madmatrix.mk) also recurses into the shared
            # directory/src/ common-library build (fixed to do so under
            # USEBUILDDIR=1 too -- it used to only clean src/ that way for
            # the non-USEBUILDDIR path), so this one call covers both.
            misc.compile(
                arg=["clean", f"BACKEND={device}", f"FPTYPE={fptype}", "USEBUILDDIR=1"],
                cwd=entry["path"],
            )
            misc.compile(
                arg=[f"BACKEND={device}", f"FPTYPE={fptype}", "USEBUILDDIR=1"], cwd=entry["path"]
            )


def regenerate_process_in_model(
    process_directory: str,
    model_spec: str,
    process_definition: str,
    rw_subdir: str = "rw_me",
    fptype: str = param_rwgt_card.DEFAULT_FPTYPE,
) -> str:
    """Regenerates process_definition in model_spec (a UFO model name, plus
    any further "import model" arguments, eg a restriction card) into
    process_directory/rw_subdir via a headless mg5_aMC run -- same subprocess-
    with-a-command-file pattern as compute_auto_widths() in launch.py --
    then compiles every resulting subprocess, at fptype, for the devices
    process_directory's own run_card.toml requests (see
    _compile_subprocesses()).

    Skips both steps if rw_subdir was already regenerated for this exact
    (model_spec, process_definition) pair (tracked via a marker file), so
    repeated reweighting runs against the same alternate model don't pay the
    regenerate cost every time -- note this means fptype only actually
    drives a compile on whichever call first regenerates a given rw_subdir;
    a later call requesting a different fptype against an already-cached
    rw_subdir gets the directory back unchanged, uncompiled for that fptype
    (run_parameter_reweighting() accounts for this itself by always calling
    _compile_subprocesses() again for the fptype it actually needs,
    regardless of whether this function regenerated anything). Returns
    rw_subdir's full path, ready to pass to Driver.load_process_directory()."""
    rw_dir = os.path.join(process_directory, rw_subdir)
    marker_path = os.path.join(rw_dir, ".madtrex_source.json")
    marker = {"model": model_spec, "process": process_definition}
    if os.path.isfile(marker_path):
        with open(marker_path) as f:
            if json.load(f) == marker:
                return rw_dir
        shutil.rmtree(rw_dir)

    mg5 = str(_MG_ROOT / "bin" / "mg5_aMC")
    if not os.path.exists(mg5):
        raise RuntimeError(
            f"cannot find mg5_aMC at {mg5}; needed to regenerate '{process_definition}' "
            f"in model '{model_spec}'"
        )
    cmds = (
        f"import model {model_spec}\n"
        f"generate {process_definition}\n"
        f"output mg7 {os.path.abspath(rw_dir)} -f\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as fh:
        fh.write(cmds)
        cmdfile = fh.name
    try:
        logger.info(
            "MadtRex: regenerating '%s' in model '%s' -> %s",
            process_definition, model_spec, rw_dir,
        )
        proc = subprocess.run([mg5, cmdfile], capture_output=False, text=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(
                f"regenerating '{process_definition}' in model '{model_spec}' failed "
                f"(mg5_aMC exited with {proc.returncode})"
            )
    finally:
        os.remove(cmdfile)

    _compile_subprocesses(rw_dir, _read_devices(process_directory), fptype)

    with open(marker_path, "w") as f:
        json.dump(marker, f)
    return rw_dir


def _sanitize_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "model"


def _tag_output_path(output_path: str, tag: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_{tag}{ext}"


def run_parameter_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
) -> dict[tuple[str | None, str], mtx.Driver]:
    """Runs MadtRex reweighting end to end against an existing MadGraph7 run.

    Parses rwgt_path (see reweight_card.py) into groups of launches by which
    UFO model *and* which floating-point precision (fptype) they target (a
    bare card with no "change model"/"set FPTYPE" lines is one group
    targeting the original process_directory as-is, at its baseline
    precision -- see reweight_card.group_by_fptype()). Each group with at
    least one launch runs as its own independent reweighting pass, since
    each needs its own MatrixElementApi loaded (a different model needs
    different physics code entirely; a different fptype needs a
    differently-compiled library for the *same* code -- see
    Driver.set_fptype_priority()): any group with a "change model" is
    regenerated/compiled into its own subdirectory of process_directory
    first (see regenerate_process_in_model()) -- named "rw_me" if the card
    only ever requests that one model, else "rw_me_<model>" per distinct
    model (shared across that model's fptype groups, since regeneration
    itself doesn't depend on fptype) -- and every group, regenerated
    directory or not, has its own fptype compiled on demand (see
    _compile_subprocesses()) before it's loaded. Each pass writes its own
    output file next to output_path (a sibling of events_path by default)
    similarly: untagged if it is the card's only group, else suffixed
    "_<model>_<fptype>" (the original-model group, if not alone, uses
    "original" for <model>). events_path is auto-discovered from the latest
    run under process_directory/Events if not given.

    The card's preamble may also set any of reweight_card.DRIVER_OPTION_NAMES
    (helicity summing, warning verbosity, event-batch splitting, deferred-
    iteration flushing, streaming -- see reweight_card.DriverOptions), applied
    to every Driver this builds, regardless of group.

    Returns a dict of the Driver built for each group, keyed by its
    (model, fptype) (model is None for the original-model group), so
    callers can inspect driver.warehouse().get() afterwards."""
    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    events_path = prepare_events(events_path)
    if output_path is None:
        output_path = default_output_path(events_path)
    init_meta = read_init_node(events_path)

    with open(rwgt_path) as f:
        parsed = param_rwgt_card.expand_scans(param_rwgt_card.parse_reweight_card(f.read()))
    # Card-wide, not per-group -- see DriverOptions -- so resolved once here
    # from the still-ungrouped ParsedCard and applied to every Driver below.
    driver_options = parsed.driver_options
    # The sample's own, actual generation-time fptype -- see
    # _read_run_fptype()'s own doc comment. Used both as group_by_fptype()'s
    # default *target* fptype (a card that sets no FPTYPE at all should stay
    # at the sample's own precision, not drift to a hardcoded default) and,
    # unconditionally below, as every group.model_spec-is-None group's
    # *normaliser* fptype -- a launch must always be normalised against this
    # original configuration, never against whichever fptype it targets.
    original_fptype = _read_run_fptype(process_directory) or param_rwgt_card.DEFAULT_FPTYPE
    groups = [
        group for group in param_rwgt_card.group_by_fptype(parsed, default_fptype=original_fptype)
        if group.launches
    ]
    if not groups:
        raise ValueError(f"reweight card {rwgt_path!r} has no 'launch' blocks")
    multi = len(groups) > 1

    devices = _read_devices(process_directory)
    drivers: dict[tuple[str | None, str], mtx.Driver] = {}
    for group in groups:
        model_tag = _sanitize_dirname(group.model_spec) if group.model_spec is not None else "original"
        tag = f"{model_tag}_{group.fptype}" if multi else None

        if group.model_spec is None:
            load_directory, group_param_card = process_directory, param_card
        else:
            rw_subdir = "rw_me" if not multi else f"rw_me_{model_tag}"
            load_directory = regenerate_process_in_model(
                process_directory, group.model_spec, group.process_definition, rw_subdir,
                fptype=group.fptype,
            )
            group_param_card = ""

        # Guarantees group.fptype is actually compiled for load_directory,
        # regardless of whether regenerate_process_in_model() just did so
        # (only true on the first group to regenerate a given rw_subdir --
        # see its own doc comment) or load_directory is process_directory
        # itself (never compiled for anything but launch.py's own
        # hardcoded default fptype at generation time).
        _compile_subprocesses(load_directory, devices, group.fptype)
        # The normaliser needs original_fptype's own compiled library too,
        # whenever it differs from the target fptype just compiled above
        # (see the driver.set_normaliser_fptype_priority() call below) --
        # only for group.model_spec is None, since that's the only case
        # normaliser_fptype_priority is actually set to something other than
        # group.fptype; a "change model" group's normaliser still shares its
        # own (regenerated) target-fptype library, unchanged.
        if group.model_spec is None and original_fptype != group.fptype:
            _compile_subprocesses(load_directory, devices, original_fptype)

        group_output_path = output_path if tag is None else _tag_output_path(output_path, tag)

        with tempfile.NamedTemporaryFile("w", suffix=".rwgt", delete=False) as fh:
            fh.write(param_rwgt_card.render_launches(group.launches))
            resolved_rwgt_path = fh.name
        try:
            logger.info(
                "MadtRex: reweighting %s against model '%s' at fptype '%s' with %s",
                events_path, group.model_spec or "<original>", group.fptype, rwgt_path,
            )
            driver = mtx.Driver()
            driver_options.apply(driver)
            driver.set_device_priority(devices)
            driver.set_fptype_priority([group.fptype])
            if group.model_spec is None:
                # Reweighting the original UFO model, possibly to a
                # different fptype: the normaliser must always reflect the
                # sample's own actual generation-time fptype
                # (original_fptype), never group.fptype -- see
                # Driver.set_normaliser_fptype_priority()'s own doc comment.
                # A no-op (normaliser_priority == fptype_priority, one api
                # shared for both roles) whenever the card doesn't actually
                # change fptype, since group.fptype then also resolves to
                # original_fptype (see group_by_fptype()'s default_fptype).
                driver.set_normaliser_fptype_priority([original_fptype])
            driver.load_process_directory(load_directory, group_param_card)
            driver.load_param_reweighting(resolved_rwgt_path)
            if init_meta is not None:
                driver.load_events(events_path, init_meta)
            else:
                driver.load_events(events_path)
            driver.warehouse().get().run()
            driver.write_weights(group_output_path)
        finally:
            os.remove(resolved_rwgt_path)
        logger.info("MadtRex: wrote %s", group_output_path)
        drivers[(group.model_spec, group.fptype)] = driver

    return drivers


def run_pdf_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
) -> mtx.PdfDriver:
    """Runs MadtRex PDF-only reweighting end to end against an existing
    MadGraph7 run, via PdfDriver -- entirely independent of run_parameter_reweighting()
    (matrix-element/parameter reweighting): see pdf_reweight_card.py's module
    docstring for why the two are split into separate cards/flows (PDFs and
    matrix elements factorise), and main() for running both together against
    the same sample.

    Unlike run_parameter_reweighting(), no model regeneration or subprocess compilation
    is needed: a PDF reweighting hypothesis is just a different (PDF set,
    scale) pair fed to the same PdfSet/PdfDriver machinery, not a different
    physics library, so this is a single pass regardless of how many "launch"
    blocks or distinct PDF sets rwgt_path uses.

    Parses rwgt_path (see pdf_reweight_card.py), first expanding any
    scan:[...] MUF/MUR/MUF_FACTOR/MUR_FACTOR values into explicit per-point
    launches (pdf_reweight_card.expand_scans(), the same mechanism
    run_parameter_reweighting() uses for physics parameter scans), then resolving the
    result into a ResolvedPdfCard: a required baseline PDF set -- unlike
    run_parameter_reweighting()'s FPTYPE, which falls back to a hard-set default when
    unspecified, MadtRex does not
    currently record which PDF set an existing sample was generated with
    anywhere *on the sample itself*. If the card's preamble never sets
    PDF_SET/BEAM1_PDF_SET/BEAM2_PDF_SET at all, this looks it up instead,
    from process_directory's own run_card.toml [beam] pdf (see
    _read_default_pdf_set()) -- a real fact about the run, not a guessed
    default, available whenever process_directory still has (or ever had)
    the Cards/ that generated events_path, which covers both "reweighting
    immediately after generation" and any later run against that same
    process directory. Only once that lookup also comes up empty (no
    run_card.toml, or one predating [beam] pdf) does pdf_rwgt_card.resolve()
    raise, requiring PDF_SET explicitly -- eg for a bare/relocated sample,
    or to deliberately reweight against a baseline other than the one the
    run actually used. A further hook could still infer the baseline PDF
    set straight from the LHE file's own metadata instead of the run_card
    -- not implemented, since MadGraph7's LHE output carries no such field
    today. Every distinct PDF set name the card references (baseline and
    launches alike) is load_pdf_set()'d once, resolved to its LHAPDF
    grid/.info files via resolve_pdf_set_files().

    Beam energies are taken from events_path's sibling header.lhe/init.lhe
    (see read_init_node()) if present, else from its LHE <init> block directly
    (PdfDriver.load_events() sniffs this itself). events_path is auto-
    discovered from the latest run under process_directory/Events if not
    given; output defaults to a sibling of events_path (see
    default_output_path()) tagged "_pdf" so it never collides with
    run_parameter_reweighting()'s own default when both run against the same sample.

    Returns the built PdfDriver, so callers can inspect driver.get()
    afterwards."""
    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    events_path = prepare_events(events_path)
    if output_path is None:
        output_path = _tag_output_path(default_output_path(events_path), "pdf")
    init_meta = read_init_node(events_path)

    with open(rwgt_path) as f:
        parsed = pdf_rwgt_card.expand_scans(pdf_rwgt_card.parse_pdf_reweight_card(f.read()))
    baseline = parsed.baseline
    if baseline.pdf_set_name is None and baseline.beam1_pdf_set_name is None and baseline.beam2_pdf_set_name is None:
        default_pdf_set = _read_default_pdf_set(process_directory)
        if default_pdf_set is not None:
            logger.info(
                "MadtRex: PDF reweight card %s sets no baseline PDF_SET; inferring '%s' "
                "from %s",
                rwgt_path, default_pdf_set, os.path.join(process_directory, "Cards", "run_card.toml"),
            )
            baseline.pdf_set_name = default_pdf_set
    card = pdf_rwgt_card.resolve(parsed)
    if not card.launches:
        raise ValueError(f"PDF reweight card {rwgt_path!r} has no 'launch' blocks")

    driver = mtx.PdfDriver()
    if card.verbose is not None:
        driver.set_verbose(card.verbose)
    if card.alpha_s_power is not None:
        driver.set_alpha_s_power(card.alpha_s_power)
    if card.scale_variation_output is not None:
        driver.set_scale_variation_output(
            getattr(mtx.ScaleVariationOutput, card.scale_variation_output.upper())
        )

    def as_scale_choice(scale: pdf_rwgt_card.PdfScale) -> mtx.ScaleChoice:
        return mtx.ScaleChoice(scale.value, scale.absolute)

    loaded_pdf_sets: set[str] = set()

    def ensure_loaded(name: str) -> None:
        if name in loaded_pdf_sets:
            return
        grid_file, alpha_s_grid_file = resolve_pdf_set_files(name)
        driver.load_pdf_set(name, grid_file, alpha_s_grid_file)
        loaded_pdf_sets.add(name)

    for name in card.baseline_pdf_set_names:
        ensure_loaded(name)
    for launch in card.launches:
        for name in launch.pdf_set_names:
            ensure_loaded(name)

    driver.set_baseline(
        *card.baseline_pdf_set_names,
        as_scale_choice(card.baseline_muF_scale),
        as_scale_choice(card.baseline_muR_scale),
    )
    for i, launch in enumerate(card.launches):
        driver.add_hypothesis(
            launch.name or f"pdf_rwgt_{i + 1}",
            *launch.pdf_set_names,
            as_scale_choice(launch.muF_scale),
            as_scale_choice(launch.muR_scale),
        )

    logger.info("MadtRex: PDF-reweighting %s with %s", events_path, rwgt_path)
    if init_meta is not None:
        # PdfDriver.load_events() only actually consults meta's beam
        # energies for binary input -- for XML LHE it reads the file's own
        # <init> block instead (older MadGraph7 writers left a zeroed-out
        # stub there, not real beam energies). set_beam_energies() is the
        # documented way to override that, so use it explicitly whenever
        # init_meta (from header.lhe/init.lhe) is available, regardless of
        # events_path's format.
        driver.set_beam_energies(init_meta.beam1_energy, init_meta.beam2_energy)
        driver.load_events(events_path, init_meta)
    else:
        driver.load_events(events_path)
    driver.get().run()
    driver.write_weights(output_path)
    logger.info("MadtRex: wrote %s", output_path)
    return driver


def _merge_reweighting_results(target, source) -> None:
    """Merges source's weight columns onto target, in place: appends every
    one of source's weight_ids after target's own, and correspondingly
    extends every target event's own wgts with the matching source event's.
    target and source must come from reweighting the exact same events, in
    the same order (true for run_generic_reweighting()'s own dispatch,
    since both passes load the same events_path) -- raises if their event
    counts differ, as a cheap sanity check of that assumption.

    Uses add_weight_id() (in-place push_back through the *existing*
    weight_ids shared_ptr) rather than reassigning target.weight_ids
    directly -- the bound weight_ids property's setter replaces the whole
    shared_ptr, which would desync every event's own (separately
    shared_ptr-aliased) weight_ids from it and corrupt the write. See
    madtrex::PdfDriver::keep_weights()'s own doc comment (pdfware.hpp) for
    the same gotcha encountered the other way round (dropping columns
    rather than adding them)."""
    target_events = target.events
    source_events = source.events
    if len(target_events) != len(source_events):
        raise ValueError(
            "cannot coalesce reweighting results: event counts differ "
            f"({len(target_events)} vs {len(source_events)}) -- were they reweighted "
            f"against the same events file?"
        )
    for name in source.weight_ids:
        target.add_weight_id(name)
    for t_ev, s_ev in zip(target_events, source_events):
        t_ev.wgts = t_ev.wgts + s_ev.wgts


def run_generic_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
) -> dict[str, object]:
    """Runs MadtRex's generic reweight card end to end: splits rwgt_path
    (see generic_reweight_card.py) into its "parameter"/"pdf" sections --
    both, or whichever one is present -- runs each through its own real
    routine (run_parameter_reweighting()/run_pdf_reweighting()) exactly as
    if it had been handed a standalone card of that kind, then, if *both*
    modes ran, coalesces the results back into a single sample per
    parameter-reweighting group: every parameter-reweighting output
    additionally gets the PDF-reweighting weights merged onto it (in place,
    then rewritten to that same group's own output path), so a caller only
    looking at that one "primary" output file sees every requested weight,
    from both modes, on one set of events -- not two separate files to
    reconcile themselves. If only one mode is present, nothing needs
    coalescing -- that mode's own output, written by its own routine, is
    already the whole answer.

    events_path/output_path/process_directory/param_card are all forwarded
    to whichever of run_parameter_reweighting()/run_pdf_reweighting()
    actually runs; param_card is unused if the card has no "parameter"
    section. output_path anchors the *parameter* pass's own naming (its
    usual untagged-unless-multiple-groups scheme, same as
    run_parameter_reweighting() alone); the PDF pass's own intermediate
    output is separately tagged "_pdf" whenever a parameter section is also
    present, mirroring main()'s own two-flag convention, so the two never
    collide.

    Returns {"parameter": dict[(model, fptype), Driver] | None, "pdf":
    PdfDriver | None} -- one real entry per mode that actually ran
    (mirroring each routine's own native return value), None for a mode
    the card didn't request. Callers after the *merged* result should read
    the output file(s) written, not these driver objects -- which of them
    ends up with its in-memory weight_ids/events mutated by the merge (the
    parameter side's, when both modes ran) is an implementation detail, not
    part of the return contract."""
    with open(rwgt_path) as f:
        sections = generic_rwgt_card.split_by_mode(f.read())
    if not sections:
        raise ValueError(f"generic reweight card {rwgt_path!r} has no 'set mode' sections")

    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    if output_path is None:
        output_path = default_output_path(events_path)

    result: dict[str, object] = {"parameter": None, "pdf": None}
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "parameter" in sections:
            param_path = os.path.join(tmp_dir, "parameter.rwgt")
            with open(param_path, "w") as f:
                f.write(sections["parameter"])
            result["parameter"] = run_parameter_reweighting(
                param_path,
                events_path=events_path,
                output_path=output_path,
                process_directory=process_directory,
                param_card=param_card,
            )
        if "pdf" in sections:
            pdf_path = os.path.join(tmp_dir, "pdf.rwgt")
            with open(pdf_path, "w") as f:
                f.write(sections["pdf"])
            pdf_output_path = _tag_output_path(output_path, "pdf") if "parameter" in sections else output_path
            result["pdf"] = run_pdf_reweighting(
                pdf_path,
                events_path=events_path,
                output_path=pdf_output_path,
                process_directory=process_directory,
            )

    param_drivers, pdf_driver = result["parameter"], result["pdf"]
    if param_drivers is not None and pdf_driver is not None:
        multi = len(param_drivers) > 1
        for (model_spec, fptype), driver in param_drivers.items():
            model_tag = _sanitize_dirname(model_spec) if model_spec is not None else "original"
            tag = f"{model_tag}_{fptype}" if multi else None
            group_output_path = output_path if tag is None else _tag_output_path(output_path, tag)
            _merge_reweighting_results(driver.warehouse().get(), pdf_driver.get())
            driver.write_weights(group_output_path)
            logger.info(
                "MadtRex: coalesced PDF reweighting into parameter reweighting result for "
                "model '%s' at fptype '%s' -> %s",
                model_spec or "<original>", fptype, group_output_path,
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MadtRex event reweighting against an existing MadGraph7 run. "
                     "Parameter (matrix-element) and PDF reweighting are independent "
                     "passes -- see run_parameter_reweighting()/run_pdf_reweighting(). Pass "
                     "--rwgt-card alone for a generic card that dispatches ('set mode "
                     "parameter'/'set mode pdf') to one or both automatically and coalesces "
                     "the result -- see run_generic_reweighting() -- or --param-rwgt-card/"
                     "--pdf-rwgt-card directly (either, or both, for two separate outputs)."
    )
    parser.add_argument(
        "--rwgt-card", dest="generic_rwgt_path", default=None,
        help="generic reweight card dispatching to parameter and/or PDF reweighting (see "
             "generic_reweight_card.py), eg Cards/reweight_card.dat; mutually exclusive "
             "with --param-rwgt-card/--pdf-rwgt-card",
    )
    parser.add_argument(
        "--param-rwgt-card", dest="param_rwgt_path", default=None,
        help="parameter (SLHA/matrix-element) reweight card, eg "
             "Cards/param_reweight_card.dat",
    )
    parser.add_argument(
        "--pdf-rwgt-card", dest="pdf_rwgt_path", default=None,
        help="PDF reweight card, eg Cards/pdf_reweight_card.dat",
    )
    parser.add_argument(
        "--events", dest="events_path", default=None,
        help="events file to reweight (events.lhe or events.npy); "
             "defaults to the latest run under Events/",
    )
    parser.add_argument(
        "--output", dest="output_path", default=None,
        help="output path, or binary weight-file prefix for events.npy input; "
             "defaults to a sibling of --events (--pdf-rwgt-card's own output, when both "
             "flags are given, is further tagged '_pdf' so the two don't collide)",
    )
    parser.add_argument(
        "--process-dir", dest="process_directory", default=".",
        help="process directory containing SubProcesses/ and Cards/ (default: .)",
    )
    parser.add_argument(
        "--param-card", dest="param_card", default="",
        help="baseline param_card.dat (--param-rwgt-card/--rwgt-card only); defaults to "
             "<process-dir>/Cards/param_card.dat",
    )
    args = parser.parse_args()
    if args.generic_rwgt_path is not None:
        if args.param_rwgt_path is not None or args.pdf_rwgt_path is not None:
            parser.error("--rwgt-card is mutually exclusive with --param-rwgt-card/--pdf-rwgt-card")
        run_generic_reweighting(
            args.generic_rwgt_path,
            events_path=args.events_path,
            output_path=args.output_path,
            process_directory=args.process_directory,
            param_card=args.param_card,
        )
        return
    if args.param_rwgt_path is None and args.pdf_rwgt_path is None:
        parser.error("at least one of --rwgt-card/--param-rwgt-card/--pdf-rwgt-card is required")
    if args.param_rwgt_path is not None:
        run_parameter_reweighting(
            args.param_rwgt_path,
            events_path=args.events_path,
            output_path=args.output_path,
            process_directory=args.process_directory,
            param_card=args.param_card,
        )
    if args.pdf_rwgt_path is not None:
        # An explicit --output is untagged by default (run_pdf_reweighting()
        # only tags "_pdf" onto its own auto-derived default); tag it
        # ourselves here so an explicit --output can't collide with
        # --param-rwgt-card's own write when both flags are given.
        pdf_output_path = args.output_path
        if pdf_output_path is not None and args.param_rwgt_path is not None:
            pdf_output_path = _tag_output_path(pdf_output_path, "pdf")
        run_pdf_reweighting(
            args.pdf_rwgt_path,
            events_path=args.events_path,
            output_path=pdf_output_path,
            process_directory=args.process_directory,
        )


if __name__ == "__main__":
    main()
