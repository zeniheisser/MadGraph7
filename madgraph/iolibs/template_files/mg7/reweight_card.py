"""Parsing/preprocessing for MadtRex reweight cards.

madspace.madtrex.ParamHandler.read_rwgt_card() (madspace/include/madtrex/
param_handler.hpp) only understands the literal "# launch / # set BLOCK_NAME
PARAM_ID PARAM_VALUE" grammar parsed by REX::tea::rwgt_slha::parse_rwgt_card
(Rex/src/teaRex.cc) -- unrecognized lines (eg "change model ...") are silently
ignored there, and scan:[...] value expressions fail to parse as a plain
double. This module extracts the extra directives MadtRex needs to act on
before handing a card to that low-level parser:

- any number of "change model <spec>" lines, each paired with a "change
  process <definition>" line, requesting the launches that follow be run
  against a different UFO model, regenerated and compiled on demand (see
  madtrex.py's regenerate_process_in_model()) -- MG7 does not persist the
  original "generate <process>" command anywhere on disk, so it must be given
  again explicitly whenever the model changes. Launches are grouped by their
  (model, process) target regardless of how they're interleaved in the card,
  so "change model" may appear more than once (switching back and forth, or
  requesting several distinct alternate models) without duplicating work;
- scan:[...] values in "set" lines, expanded into explicit per-value launch
  blocks via the same eval-based mechanism and group-index semantics as
  models.check_param_card.ParamCardIterator.iterate().
"""

import itertools
import re
from dataclasses import dataclass, field

_MODEL_RE = re.compile(r"^\s*change\s+model\s+(?P<spec>.+?)\s*$", re.I)
_PROCESS_RE = re.compile(r"^\s*change\s+process\s+(?P<definition>.+?)\s*$", re.I)
_LAUNCH_RE = re.compile(r"^\s*launch\b(?P<rest>.*)$", re.I)
_NAME_RE = re.compile(r"rwgt_name\s*=\s*(?P<name>\S+)", re.I)
_SET_RE = re.compile(
    r"^\s*set\s+(?:param_card\s+)?(?P<block>\S+)\s+(?P<id>\S+)\s+(?P<value>.+?)\s*$",
    re.I,
)
_SCAN_RE = re.compile(r"scan\s*(?P<group>\d*)\s*:\s*(?P<values>.*)$", re.I)


@dataclass
class LaunchBlock:
    name: str | None
    sets: list[tuple[str, str, str]] = field(default_factory=list)  # (block, id, value)


@dataclass
class ModelGroup:
    # model_spec is None for launches that never followed a "change model"
    # line, ie ones meant to run against the original process directory as-is.
    model_spec: str | None
    process_definition: str | None
    launches: list[LaunchBlock] = field(default_factory=list)


@dataclass
class ParsedCard:
    # One entry per distinct (model_spec, process_definition) target
    # actually used by a launch, in order of first appearance.
    groups: list[ModelGroup]


def parse_reweight_card(text: str) -> ParsedCard:
    """Splits text into per-(model, process)-target groups of "launch"/"set"
    blocks. Lines are matched loosely (case-insensitive, "#" strips trailing
    comments) to stay tolerant of the same card conventions the legacy
    reweight_card.dat format allows (eg "set param_card BLOCK ID VALUE
    # orig: ...").

    Raises if a "launch" follows a "change model" with no paired "change
    process" yet (MG7 has no other way to learn the process definition), or if
    the same model is later paired with a different process definition (each
    model may be regenerated for at most one process per card)."""
    groups: dict[str | None, ModelGroup] = {}
    order: list[str | None] = []
    current_model_spec: str | None = None
    current_process_definition: str | None = None
    current_group: ModelGroup | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]
        if not line.strip():
            continue

        model_match = _MODEL_RE.match(line)
        if model_match:
            current_model_spec = model_match.group("spec").strip()
            continue

        process_match = _PROCESS_RE.match(line)
        if process_match:
            current_process_definition = process_match.group("definition").strip()
            continue

        launch_match = _LAUNCH_RE.match(line)
        if launch_match:
            if current_model_spec is not None and current_process_definition is None:
                raise ValueError(
                    f"reweight card requests 'change model {current_model_spec}' but "
                    f"has no 'change process <definition>' line before the next "
                    f"'launch': MG7 does not persist the original process definition "
                    f"anywhere on disk, so it must be given again explicitly to "
                    f"regenerate it in the new model"
                )
            group = groups.get(current_model_spec)
            if group is None:
                group = ModelGroup(current_model_spec, current_process_definition)
                groups[current_model_spec] = group
                order.append(current_model_spec)
            elif current_model_spec is not None and group.process_definition != current_process_definition:
                raise ValueError(
                    f"reweight card requests model '{current_model_spec}' with two "
                    f"different process definitions ('{group.process_definition}' and "
                    f"'{current_process_definition}'); MadtRex only supports "
                    f"regenerating a single process per model per card"
                )
            name_match = _NAME_RE.search(launch_match.group("rest"))
            group.launches.append(LaunchBlock(name=name_match.group("name") if name_match else None))
            current_group = group
            continue

        set_match = _SET_RE.match(line)
        if set_match:
            if current_group is None or not current_group.launches:
                raise ValueError(f"reweight card 'set' line appears before any 'launch': {raw_line!r}")
            current_group.launches[-1].sets.append(
                (set_match.group("block"), set_match.group("id"), set_match.group("value"))
            )
            continue

    return ParsedCard(groups=[groups[key] for key in order])


def _expand_launch(block: LaunchBlock, base_index: int) -> list[LaunchBlock]:
    """Expands one LaunchBlock's scan:[...] values (if any) into the Cartesian
    product of concrete-valued LaunchBlocks. "set" lines sharing the same
    scanN: group index vary together; ungrouped ("scan:") entries each vary
    independently -- mirrors models.check_param_card.ParamCardIterator.iterate()."""
    groups: dict[object, list[int]] = {}
    values_by_index: dict[int, list] = {}
    for i, (_, _, raw_value) in enumerate(block.sets):
        value = raw_value.strip()
        if not value.lower().startswith("scan"):
            continue
        scan_match = _SCAN_RE.match(value)
        if not scan_match:
            continue
        try:
            values_by_index[i] = eval(scan_match.group("values"))
        except SyntaxError as error:
            raise ValueError(f"invalid scan definition {raw_value!r}: {error}") from error
        group = scan_match.group("group")
        key = group if group else object()  # ungrouped: unique key, varies independently
        groups.setdefault(key, []).append(i)

    if not groups:
        return [block]

    keys = list(groups.keys())
    lengths = [range(len(values_by_index[groups[key][0]])) for key in keys]
    name = block.name or f"rwgt_{base_index}"
    expanded = []
    for point, positions in enumerate(itertools.product(*lengths)):
        sets = list(block.sets)
        for key, pos in zip(keys, positions):
            for i in groups[key]:
                block_name, param_id, _ = block.sets[i]
                sets[i] = (block_name, param_id, repr(values_by_index[i][pos]))
        expanded.append(LaunchBlock(name=f"{name}_{point + 1}", sets=sets))
    return expanded


def expand_scans(card: ParsedCard) -> ParsedCard:
    """Returns a copy of card with every launch's scan:[...] values resolved
    into explicit per-point launch blocks, within each model group
    independently; a no-op for cards with no scans."""
    groups = []
    for group in card.groups:
        launches = []
        for i, block in enumerate(group.launches):
            launches.extend(_expand_launch(block, i + 1))
        groups.append(ModelGroup(group.model_spec, group.process_definition, launches))
    return ParsedCard(groups=groups)


def render_launches(launches: list[LaunchBlock]) -> str:
    """Renders one group's launches back into the literal "launch"/
    "set BLOCK ID VALUE" text ParamHandler.read_rwgt_card() understands -- no
    "change model"/"change process" lines and no scan:[...] values, both of
    which must already be resolved (see expand_scans() and the model-group
    handling in madtrex.py)."""
    lines = []
    for block in launches:
        lines.append(f"launch --rwgt_name={block.name}" if block.name else "launch")
        for block_name, param_id, value in block.sets:
            lines.append(f"set {block_name} {param_id} {value}")
    return "\n".join(lines) + "\n"
