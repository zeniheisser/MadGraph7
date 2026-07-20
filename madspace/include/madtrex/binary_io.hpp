// MadtRex binary I/O: builds/writes REX::lhe objects against MadGraph7's
// internal binary event-file format (madspace::EventFile, io.hpp) instead of
// LHEF XML, without teaching Rex itself anything about that format.
//
// Rex already factors "assemble/sort a REX::lhe from some raw format" into a
// generic REX::lheReader<InitRaw, EventRaw, HeaderRaw>/REX::lheWriter<...>
// pair driven by plain translator functions (see Rex.h); REX::xml_reader()/
// xml_writer() are simply that template instantiated with shared_ptr<xmlNode>
// as the raw type. This header instantiates the same machinery with
// madspace::LHEMeta/LHEEvent (io.hpp's lhe_output.hpp raw structs, which are
// already field-for-field identical to the binary format's
// EventRecord::f_lhe_event/ParticleRecord::f_lhe_particle layout) as the raw
// types, so all translation/sorting bookkeeping is still Rex's, and only the
// binary<->LHEEvent/LHEMeta plumbing lives here.

#pragma once

#include "tupperware.hpp"

#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"

#include <memory>
#include <string>

namespace madtrex {

// Reads a MadGraph7 internal binary event file (an EventFile opened in
// EventFile::load mode, expected to have been written with
// EventRecord::f_lhe_event | ParticleRecord::f_lhe_particle layout, eg via
// EventGenerator::combine_to_lhe_npy) into a REX::lhe.
//
// The binary format itself carries no init-block information (beam ids/
// energies, PDF set, per-process cross sections): MadGraph7's own driver
// constructs that separately (see madspace::LHEMeta) and does not persist it
// alongside the binary event file. meta is therefore an optional, caller-
// supplied source for that information; the resulting REX::lhe's initNode is
// left at its default (all-zero) values for whichever fields meta doesn't
// specify. This has no effect on reweighting itself: procReweightor::
// evaluate() only ever reads per-event data (momenta, pdg, status, flavor_/
// helicity_ indices, alphaS), never initNode fields.
std::shared_ptr<REX::lhe> load_lhe_binary(
    const std::string& path, const madspace::LHEMeta& meta = {}
);
// As above, given an already-open EventFile (must be in EventFile::load
// mode, with a layout compatible with EventRecord::f_lhe_event |
// ParticleRecord::f_lhe_particle).
std::shared_ptr<REX::lhe> load_lhe_binary(
    madspace::EventFile& file, const madspace::LHEMeta& meta = {}
);

// Writes the per-event weights recorded in lhe.events[i]->wgts_ (ie the
// output of running a REX::tea::reweightor over lhe) back out as one
// madspace weight-only binary file per entry of lhe.weight_ids, using
// madspace::weight_file_layout (EventRecord::f_weight, no particle data) --
// the same index-aligned, kinematics-free weight-file convention already
// used for per-channel weights in ChannelEventGenerator. Files are named
// "<path_prefix>_<weight_id>.npy"; weight_id is sanitized to a
// filesystem-safe token if needed. Throws if an event's wgts_ is shorter
// than lhe.weight_ids (ie the reweighting loop didn't populate that weight
// id for every event).
void save_weights_binary(const REX::lhe& lhe, const std::string& path_prefix);

} // namespace madtrex
