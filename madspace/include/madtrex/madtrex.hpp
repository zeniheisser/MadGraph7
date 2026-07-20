/* ################################################### */
/* #                                                 # */
/* #     ___  ___          _ _  ______               # */
/* #     |  \/  |         | | | | ___ \              # */
/* #     | .  . | __ _  __| | |_| |_/ /_____  __     # */
/* #     | |\/| |/ _` |/ _` | __|    // _ \ \/ /     # */
/* #     | |  | | (_| | (_| | |_| |\ \  __/>  <      # */
/* #     \_|  |_/\__,_|\__,_|\__\_| \_\___/_/\_\     # */
/* #                                                 # */
/* #    ******************************************   # */
/* #    * MadGraph7 teaRex Reweighting executors *   # */
/* #    ******************************************   # */
/* ################################################### */
//
// MadtRex is the primary MadGraph7 driver for data-parallel
// event reweighting, using the Rex/teaRex libraries wrapper
// with the TupperWare and WareHouse classes to simplify the
// interface between MadGraph7 and Rex/teaRex.
// MadtRex provides immediate access to SLHA parameter rewegighting
// with UMAMI-compliant matrix element libraries through the 
// MadtRex driver type, which can load data directly from
// provided paths to relevant process information, i.e. LHE
// files, reweighting cards, SLHA parameter cards, and the
// utilised scattering amplitude libraries.
// The driver is minimal, only wrapping the REX::tea::reweightor
// and TupperWare types with necessary plumbing and bureaucracy
// to utilise scattering amplitudes safely.

#pragma once
#include "binary_io.hpp"
#include "param_handler.hpp"
#include "tupperware.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace madtrex {

using namespace madspace;

    // Which on-disk representation a set of reweighting events came from, or
    // should be written back out as: LHEF XML, or MadGraph7's default
    // internal binary EventFile format (io.hpp).
    enum class LheFormat { xml, binary };

    // Peeks at path's first bytes to tell a madspace binary EventFile (NumPy
    // magic "\x93NUMPY", see io.cpp's write_event_header) apart from LHEF
    // XML, without relying on the file extension. Throws if path cannot be
    // opened.
    LheFormat detect_lhe_format(const std::string& path);

    std::shared_ptr<REX::lhe> load_lhe_xml(const std::string& path);
    std::shared_ptr<REX::lhe> load_lhe_xml(std::istream& stream);
    // Dispatches to load_lhe_xml or madtrex::load_lhe_binary based on
    // detect_lhe_format(path).
    std::shared_ptr<REX::lhe> load_lhe(const std::string& path);

    // ---------------------------------------------------------------------
    // MadMatrix/UMAMI subprocess JSON loading
    // ---------------------------------------------------------------------
    //
    // MadMatrix compiles each physical subprocess of a MadGraph7 process (eg
    // the "gg > ttx" piece of "p p > 2j") into its own UMAMI-compliant matrix
    // element library, and describes the resulting set of subprocesses in a
    // PROCESS_DIRECTORY/SubProcesses/subprocesses.json file: a JSON array with
    // one entry per subprocess, giving its incoming/outgoing PDG codes, the
    // (device-templated) path to its compiled library, the flavor channels it
    // covers (eg one library often serves several external quark flavors, up
    // to the color/coupling factors UMAMI computes internally), and the full
    // list of helicity combinations it was generated for.
    //
    // The types and functions below parse that file and turn it into
    // REX::tea-ready building blocks: SubProcessSpec is the parsed form of one
    // entry, make_event_checker() builds the flavor/helicity-aware
    // REX::eventBelongs for it, resolve_me_path() finds the concrete library
    // file the device-templated me_path refers to, and MadtRex ties all of
    // this together with a madspace::Context and a madtrex::WareHouse into a
    // single driver for one process directory.

    // One entry of a subprocess's JSON "flavors" array, expanded so that
    // mirrored entries (incoming beams swapped, eg q qbar vs qbar q sharing one
    // library) become a second, explicit FlavorChannel instead of needing
    // special-casing at match time. `pdgs` is the exact, positional external-leg
    // PDG tuple expected of an event's own particle slot order (incoming legs,
    // in event order, followed by outgoing legs, in event order) for this
    // specific variant; `mirrored` records whether this variant is the
    // incoming-swapped copy of a JSON option, so per-event helicity lookups
    // know to undo the swap before consulting SubProcessSpec::helicities, which
    // is always recorded in the un-mirrored leg order.
    struct FlavorChannel {
        int index = -1;
        std::vector<long int> pdgs;
        bool mirrored = false;
    };

    // Parsed form of one entry of a subprocesses.json array: one physical
    // MadMatrix subprocess, carrying enough information to load its matrix
    // element library and build a flavor/helicity-aware event checker for it.
    // Phase-space-only JSON fields (channels, color_flows, pdg_color_types) are
    // intentionally not retained here: they describe multichannel phase-space
    // generation, not the reweighting-facing contract between an event and a
    // UMAMI matrix element call.
    struct SubProcessSpec {
        std::vector<long int> incoming;
        std::vector<long int> outgoing;
        std::string me_path_template; // eg "lib/libmadmatrix_P1_gg_ttx_{device}.so", relative to PROCESS_DIRECTORY
        std::string source_path;      // JSON "path" field; informational only, may be stale if the process directory was moved since generation
        std::size_t diagram_count = 0;
        std::vector<FlavorChannel> flavors;       // expanded, incl. mirror variants
        std::vector<std::vector<int>> helicities; // as given in JSON, un-mirrored leg order
    };

    // Parses a MadMatrix subprocesses.json file (eg
    // PROCESS_DIRECTORY/SubProcesses/subprocesses.json) into one
    // SubProcessSpec per top-level array entry.
    std::vector<SubProcessSpec> parse_subprocesses_json(const std::string& subprocesses_json_path);

    // Builds the flavor/helicity-aware REX::eventBelongs for a single
    // SubProcessSpec: one template event per distinct (pre-mirror-expansion)
    // flavor option, matched via the default external_legs_comparator (ie PDG
    // multiset per status, which mirroring doesn't change), with
    // flavor_index/helicity_index hash functions that positionally re-match
    // the specific (possibly mirrored) variant to report the UMAMI flavor
    // index and, if the event carries genuine per-particle helicities in its
    // spin_ fields (LHEF-style +-1; anything else is treated as "no fixed
    // helicity"), the position of that helicity combination in
    // SubProcessSpec::helicities. Events without fixed helicities are left
    // with helicity_ unset (REX::npos truncated into event::helicity_'s -1
    // sentinel), meaning the matrix element should sum over helicities.
    REX::eventBelongs make_event_checker(const SubProcessSpec& spec);

    // Given PROCESS_DIRECTORY and a me_path template containing a literal
    // "{device}" placeholder, lists whichever concrete files actually exist on
    // disk (eg "libmadmatrix_P1_gg_ttx_cppnone.so") and resolves which one to
    // use via device_priority: the first priority entry with a matching file
    // wins; if none of the priority entries match but exactly one file was
    // found, that one is used regardless; otherwise (no match, or several
    // found and no priority set) throws, listing what was found, so the caller
    // can call MadtRex::set_device_priority(...) and try again.
    std::string resolve_me_path(
        const std::string& process_directory,
        const std::string& me_path_template,
        const std::vector<std::string>& device_priority
    );

    // Driver is the top-level driver tying together JSON-described MadMatrix
    // subprocesses, the madspace::Context that loads and owns their UMAMI
    // matrix element libraries, and the madtrex::WareHouse of
    // REX::tea::procReweightors built from them. One Driver instance
    // corresponds to one MadGraph7 process directory (eg PROCMG7_sm_2):
    // load_process_directory()/load_process() may be called more than once to
    // register subprocesses from several subprocesses.json files into the same
    // WareHouse before calling load_events() (or warehouse().build(lhe)
    // directly, if the events are already in hand as a REX::lhe rather than
    // on disk).
    class Driver {
    public:
        Driver();
        explicit Driver(madspace::ContextPtr context);

        // Priority order used to disambiguate multiple {device}-suffixed
        // library variants found for the same subprocess (eg prefer
        // "cppavx2" over "cppnone" once vectorized kernels exist). Earlier
        // entries win. Empty by default, meaning: auto-use a lone match,
        // throw on ambiguity.
        Driver& set_device_priority(std::vector<std::string> priority);
        const std::vector<std::string>& device_priority() const;

        // Parses subprocesses_json_path, loads each entry's matrix element
        // library (resolving {device} against what's actually on disk) via
        // the owned Context, and registers one TupperWare (api +
        // flavor/helicity-aware event checker) per entry into the owned
        // WareHouse. param_card defaults to
        // PROCESS_DIRECTORY/Cards/param_card.dat, where PROCESS_DIRECTORY is
        // taken to be the grandparent directory of subprocesses_json_path (ie
        // .../PROCESS_DIRECTORY/SubProcesses/subprocesses.json).
        Driver& load_process(const std::string& subprocesses_json_path, const std::string& param_card = "");
        // Convenience: process_directory/SubProcesses/subprocesses.json.
        Driver& load_process_directory(const std::string& process_directory, const std::string& param_card = "");

        // Configures UMAMI-direct parameter reweighting: builds (or
        // replaces) the owned ParamHandler over every matrix element api
        // registered so far (ie everything load_process()/
        // load_process_directory() has loaded up to this call -- call this
        // after those, not before), parses rwgt_path, and wires the
        // resulting iterators/launch_names onto the WareHouse via
        // warehouse().set_iterators()/set_launch_names(). Must be called
        // before load_events() (or warehouse().build()), since that is what
        // actually captures the WareHouse's current iterators/launch_names
        // into the built reweightor. Requires every registered api to
        // implement umami_set_parameter (see ParamHandler::add_api); throws
        // otherwise -- for libraries that don't, build the iterators from
        // REX::tea::param_rwgt's file-based approach instead and pass them to
        // warehouse().set_iterators() directly.
        Driver& load_param_reweighting(const std::string& rwgt_path);

        ParamHandler& param_handler();
        const ParamHandler& param_handler() const;

        // Loads the event source the reweighting will run against: sniffs
        // path via detect_lhe_format() to decide between MadGraph7's
        // internal binary EventFile format and LHEF XML, loads it
        // accordingly, and builds the WareHouse's REX::tea::reweightor
        // against the result (ie warehouse().build(*load_lhe(path))).
        // meta supplies init-block information (beam ids/energies, xsec,
        // PDF set) for the binary path, which -- unlike LHEF -- stores none
        // of that on disk; ignored when path is XML, whose own <init> block
        // is used instead. Also sets binary_output() to match the detected
        // input format, so write_weights() round-trips to the same
        // representation by default; call set_binary_output() afterwards to
        // override. May be called again to rebuild against a different
        // event source, discarding any previously-built reweightor (see
        // WareHouse::build()).
        Driver& load_events(const std::string& path, const madspace::LHEMeta& meta = {});

        // Which format write_weights() targets: true for
        // madtrex::save_weights_binary() (one weight-only binary EventFile
        // per weight id), false for a full LHEF XML rewrite via
        // REX::write_lhef() (events plus every recorded weight). Defaults to
        // whatever load_events() last detected; false if load_events() was
        // never called.
        bool binary_output() const;
        Driver& set_binary_output(bool binary);

        // Writes out whatever weights are currently recorded on the built
        // reweightor's events (ie warehouse().get(), typically after
        // REX::tea::reweightor::run()) via save_weights_binary(path) or
        // REX::write_lhef(path), according to binary_output(). Throws if
        // the WareHouse hasn't been built yet (see load_events()).
        Driver& write_weights(const std::string& path);

        // End-to-end convenience running the entire pipeline in one call:
        // load_process_directory(process_directory, param_card), then --
        // if rwgt_path is non-empty -- load_param_reweighting(rwgt_path),
        // then load_events(events_path, meta), then
        // warehouse().get().run() (REX::tea::reweightor's full setup/
        // run_all_iterations/finalise_reweighting loop), then
        // write_weights(output_path). rwgt_path may be left empty to skip
        // parameter reweighting (eg when every process's WareHouse entry
        // already fully determines its own reweighting via
        // add_api()/set_normaliser() and needs no per-iteration parameter
        // changes at all). Returns *this so eg driver.reweight(...).warehouse()
        // stays chainable for inspecting the result afterwards.
        Driver& reweight(
            const std::string& process_directory,
            const std::string& events_path,
            const std::string& output_path,
            const std::string& rwgt_path = "",
            const std::string& param_card = "",
            const madspace::LHEMeta& meta = {}
        );

        madspace::ContextPtr context() const;
        WareHouse& warehouse();
        const WareHouse& warehouse() const;
        const std::vector<SubProcessSpec>& specs() const;

    private:
        madspace::ContextPtr _context;
        std::vector<std::string> _device_priority = {"cpp512z", "cpp512y", "cppavx2", "cppsse4", "cppnone"};
        WareHouse _warehouse;
        std::vector<SubProcessSpec> _specs;
        ParamHandler _param_handler;
        bool _binary_output = false;
    };

}