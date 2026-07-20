import argparse
import glob
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
# madevent.py resolves. Unlike madevent.py, this module does not trigger the
# installer itself: by the time reweighting is needed, generate_events has
# already ensured the install exists.
import madgraph as _mg_pkg
_MG_ROOT = Path(_mg_pkg.__file__).parents[1]
_INSTALL_DIR = _MG_ROOT / "madspace" / "install"
if (_INSTALL_DIR / "madspace").is_dir() and str(_INSTALL_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALL_DIR))

import madspace.madtrex as mtx
from madgraph.various import misc
from madgraph.various.banner import RunCardMG7

from . import reweight_card as rwgt_card

logger = logging.getLogger("madevent7")


def find_latest_run_dir(events_dir: str = "Events") -> str:
    """Returns the most recently modified Events/<run_name>_NN run directory."""
    run_dirs = glob.glob(os.path.join(events_dir, "*_[0-9][0-9]"))
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {events_dir}")
    return max(run_dirs, key=os.path.getmtime)


def find_run_events(run_dir: str) -> str:
    """Returns run_dir's generated events file, preferring LHEF XML over the
    internal binary format (see madevent.py's generate_events, which writes
    events.lhe for output_format="lhe" and events.npy for "lhe_npy")."""
    for name in ("events.lhe", "events.npy"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no events.lhe or events.npy found in {run_dir}")


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
    run_card = RunCardMG7(os.path.join(process_directory, "Cards", "run_card.toml"))
    devices = run_card["run"]["devices"]
    return devices if isinstance(devices, list) else [devices]


def regenerate_process_in_model(
    process_directory: str,
    model_spec: str,
    process_definition: str,
    rw_subdir: str = "rw_me",
) -> str:
    """Regenerates process_definition in model_spec (a UFO model name, plus
    any further "import model" arguments, eg a restriction card) into
    process_directory/rw_subdir via a headless mg5_aMC run -- same subprocess-
    with-a-command-file pattern as compute_auto_widths() in madevent.py --
    then compiles every resulting subprocess for the devices
    process_directory's own run_card.toml requests, mirroring the lazy
    per-device compile in madevent.py's MadgraphSubprocess.__init__.

    Skips both steps if rw_subdir was already regenerated for this exact
    (model_spec, process_definition) pair (tracked via a marker file), so
    repeated reweighting runs against the same alternate model don't pay the
    regenerate+compile cost every time. Returns rw_subdir's full path, ready
    to pass to Driver.load_process_directory()."""
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
        proc = subprocess.run([mg5, cmdfile])
        if proc.returncode != 0:
            raise RuntimeError(
                f"regenerating '{process_definition}' in model '{model_spec}' failed "
                f"(mg5_aMC exited with {proc.returncode})"
            )
    finally:
        os.remove(cmdfile)

    devices = _read_devices(process_directory)
    with open(os.path.join(rw_dir, "SubProcesses", "subprocesses.json")) as f:
        subprocesses = json.load(f)
    for entry in subprocesses:
        for device in devices:
            logger.info("MadtRex: compiling %s for device '%s'", entry["path"], device)
            misc.compile(arg=[f"BACKEND={device}", "USEBUILDDIR=1"], cwd=entry["path"])

    with open(marker_path, "w") as f:
        json.dump(marker, f)
    return rw_dir


def _sanitize_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "model"


def _tag_output_path(output_path: str, tag: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_{tag}{ext}"


def run_reweighting(
    rwgt_path: str,
    events_path: str | None = None,
    output_path: str | None = None,
    process_directory: str = ".",
    param_card: str = "",
) -> dict[str | None, mtx.Driver]:
    """Runs MadtRex reweighting end to end against an existing MadGraph7 run.

    Parses rwgt_path (see reweight_card.py) into groups of launches by which
    UFO model they target (a bare card with no "change model" line is one
    group targeting the original process_directory as-is). Each group with at
    least one launch runs as its own independent reweighting pass: any group
    with a "change model" is regenerated/compiled into its own subdirectory
    of process_directory first (see regenerate_process_in_model()) -- named
    "rw_me" if it is the card's only group, else "rw_me_<model>" per distinct
    model, to keep multiple alternate-model runs apart. Each pass writes its
    own output file next to output_path (a sibling of events_path by default)
    similarly: untagged if it is the card's only group, else suffixed
    "_<model>" (the original-model group, if not alone, is suffixed
    "_original"). events_path is auto-discovered from the latest run under
    process_directory/Events if not given.

    Returns a dict of the Driver built for each group, keyed by its model
    (None for the original-model group), so callers can inspect
    driver.warehouse().get() afterwards."""
    if events_path is None:
        events_path = find_run_events(
            find_latest_run_dir(os.path.join(process_directory, "Events"))
        )
    if output_path is None:
        output_path = default_output_path(events_path)

    with open(rwgt_path) as f:
        parsed = rwgt_card.expand_scans(rwgt_card.parse_reweight_card(f.read()))
    groups = [group for group in parsed.groups if group.launches]
    if not groups:
        raise ValueError(f"reweight card {rwgt_path!r} has no 'launch' blocks")
    multi = len(groups) > 1

    devices = _read_devices(process_directory)
    drivers: dict[str | None, mtx.Driver] = {}
    for group in groups:
        tag = None
        if multi:
            tag = _sanitize_dirname(group.model_spec) if group.model_spec is not None else "original"

        if group.model_spec is None:
            load_directory, group_param_card = process_directory, param_card
        else:
            rw_subdir = "rw_me" if tag is None else f"rw_me_{tag}"
            load_directory = regenerate_process_in_model(
                process_directory, group.model_spec, group.process_definition, rw_subdir
            )
            group_param_card = ""

        group_output_path = output_path if tag is None else _tag_output_path(output_path, tag)

        with tempfile.NamedTemporaryFile("w", suffix=".rwgt", delete=False) as fh:
            fh.write(rwgt_card.render_launches(group.launches))
            resolved_rwgt_path = fh.name
        try:
            logger.info(
                "MadtRex: reweighting %s against model '%s' with %s",
                events_path, group.model_spec or "<original>", rwgt_path,
            )
            driver = mtx.Driver()
            driver.set_device_priority(devices)
            driver.load_process_directory(load_directory, group_param_card)
            driver.load_param_reweighting(resolved_rwgt_path)
            driver.load_events(events_path)
            driver.warehouse().get().run()
            driver.write_weights(group_output_path)
        finally:
            os.remove(resolved_rwgt_path)
        logger.info("MadtRex: wrote %s", group_output_path)
        drivers[group.model_spec] = driver

    return drivers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MadtRex event reweighting against an existing MadGraph7 run."
    )
    parser.add_argument(
        "--rwgt-card", required=True, dest="rwgt_path",
        help="reweight card (SLHA parameter reweighting), eg Cards/reweight_card.dat",
    )
    parser.add_argument(
        "--events", dest="events_path", default=None,
        help="events file to reweight (events.lhe or events.npy); "
             "defaults to the latest run under Events/",
    )
    parser.add_argument(
        "--output", dest="output_path", default=None,
        help="output path, or binary weight-file prefix for events.npy input; "
             "defaults to a sibling of --events",
    )
    parser.add_argument(
        "--process-dir", dest="process_directory", default=".",
        help="process directory containing SubProcesses/ and Cards/ (default: .)",
    )
    parser.add_argument(
        "--param-card", dest="param_card", default="",
        help="baseline param_card.dat; defaults to <process-dir>/Cards/param_card.dat",
    )
    args = parser.parse_args()
    run_reweighting(
        args.rwgt_path,
        events_path=args.events_path,
        output_path=args.output_path,
        process_directory=args.process_directory,
        param_card=args.param_card,
    )


if __name__ == "__main__":
    main()
