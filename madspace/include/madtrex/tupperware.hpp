/* .---------------------------------------------------------. */
/* | _____                          __        __             | */
/* ||_   _|   _ _ __  _ __   ___ _ _\ \      / /_ _ _ __ ___ | */
/* |  | || | | | '_ \| '_ \ / _ \ '__\ \ /\ / / _` | '__/ _ \| */
/* |  | || |_| | |_) | |_) |  __/ |   \ V  V / (_| | | |  __/| */
/* |  |_| \__,_| .__/| .__/ \___|_|    \_/\_/ \__,_|_|  \___|| */
/* |           |_|   |_|                                     | */
/* '---------------------------------------------------------' */
/*       Tearex-Umami Pointer Passing Event Reweighting        */
/*           With Accelerated Repetitive Execution             */
//
// TupperWare is the "simple" generic reweighting bridge between MadGraph7 and
// the standalone Rex/teaRex libraries: it turns already-loaded, UMAMI-compliant
// madspace::MatrixElementApi objects into REX::tea::weightors, ie plain
// std::function<shared_ptr<vector<double>>(REX::process&)> callables, and combines
// them with a caller-supplied event checker into a REX::tea::procReweightor.
//

#pragma once

#include "rex/teaRex.h"
#include "madspace/driver/context.hpp"

#include <memory>
#include <vector>

namespace madtrex {

// Converts an already-loaded madspace::MatrixElementApi into a REX::tea::weightor.
// Input keys are auto-derived from api.required_inputs()/supported_inputs(),
// restricted to what can actually be sourced from a REX::process: momenta
// (UMAMI_IN_MOMENTA), the strong coupling (UMAMI_IN_ALPHA_S), and, if the api
// supports channel pinning, the per-event flavor_/helicity_ indices
// (UMAMI_IN_FLAVOR_INDEX / UMAMI_IN_HELICITY_INDEX). Throws if the api requires an
// input that cannot be derived this way (eg a sampling-time random number), or if
// it does not report supporting UMAMI_OUT_MATRIX_ELEMENT. The returned weightor
// reads the process's momenta out of its flat SoA storage and reshapes them into
// UMAMI's batch-contiguous call convention; all matching events in a process are
// evaluated in a single UMAMI call.
std::shared_ptr<REX::tea::weightor> make_weightor(const madspace::MatrixElementApi& api);

// As above, but with explicit control over which UMAMI keys are used. input_keys
// may only contain keys derivable from a REX::process (UMAMI_IN_MOMENTA,
// UMAMI_IN_ALPHA_S, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_HELICITY_INDEX); output_keys
// may only contain UMAMI_OUT_MATRIX_ELEMENT and UMAMI_OUT_DIAGRAM_AMP2 (the latter
// is summed over diagrams before being folded into the result). result_keys picks
// which of the requested output_keys are summed together into the weight returned
// per event, and must be a subset of output_keys.
std::shared_ptr<REX::tea::weightor> make_weightor(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys
);

// The auto-derived key sets used by the single-argument make_weightor overload,
// exposed so callers building a manual override can start from the defaults.
// Throws under the same conditions as the single-argument make_weightor overload.
std::vector<UmamiInputKey> default_input_keys(const madspace::MatrixElementApi& api);
std::vector<UmamiOutputKey> default_output_keys(const madspace::MatrixElementApi& api);

// TupperWare bundles what's needed to build a REX::tea::procReweightor for a single
// physical process: one or more externally-owned MatrixElementApis (evaluated in
// registration order as procReweightor::reweight_functions), an optional separate
// normalisation MatrixElementApi, and an event checker defining which events belong
// to this process. The MatrixElementApis are expected to outlive the TupperWare
// (and the procReweightor/weightors it produces), matching how madspace::Context
// owns them.
class TupperWare {
public:
    TupperWare() = default;

    // Convenience constructors for the common case of a single MatrixElementApi and
    // event checker. The api is registered as the first reweight_function, and also
    // as the normaliser if no separate normaliser is set later.
    TupperWare(const madspace::MatrixElementApi& api, REX::eventBelongs checker);
    TupperWare(const madspace::MatrixElementApi& api, REX::event_bool_fn checker);

    // Registers a matrix element as the next reweight_functions amplitude
    // (registration order determines the amp index used by
    // procReweightor::evaluate(amp)), with auto-derived UMAMI keys.
    TupperWare& add_api(const madspace::MatrixElementApi& api);
    // As above, with an explicit override of the UMAMI input/output keys used for
    // this particular api instead of auto-derivation.
    TupperWare& add_api(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys
    );

    TupperWare& add_api(std::vector<const madspace::MatrixElementApi*> apis);
    TupperWare& add_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<UmamiInputKey> input_keys_list,
        std::vector<UmamiOutputKey> output_keys_list
    );
    TupperWare& add_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<std::vector<UmamiInputKey>> input_keys_list,
        std::vector<std::vector<UmamiOutputKey>> output_keys_list
    );

    TupperWare& set_api(std::vector<const madspace::MatrixElementApi*> apis);
    TupperWare& set_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<UmamiInputKey> input_keys_list,
        std::vector<UmamiOutputKey> output_keys_list
    );
    TupperWare& set_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<std::vector<UmamiInputKey>> input_keys_list,
        std::vector<std::vector<UmamiOutputKey>> output_keys_list
    );

    // Registers the matrix element used for normalisation
    // (procReweightor::normaliser). If never called, build() leaves the normaliser
    // unset, which makes procReweightor::initialise() default to the first
    // reweight function.
    TupperWare& set_normaliser(const madspace::MatrixElementApi& api);
    TupperWare& set_normaliser(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys
    );

    // Which UMAMI output key(s) are summed into the weight returned by every
    // weightor generated by this TupperWare, including the normaliser's. Defaults
    // to just UMAMI_OUT_MATRIX_ELEMENT.
    TupperWare& set_result_keys(std::vector<UmamiOutputKey> keys);

    // Defines which events belong to this process. Setting one clears the other.
    TupperWare& set_event_checker(REX::eventBelongs checker);
    TupperWare& set_event_checker(std::shared_ptr<REX::eventBelongs> checker);
    TupperWare& set_event_checker(REX::event_bool_fn checker);

    // Builds the procReweightor from the registered apis and event checker.
    // Throws if no apis or no event checker have been registered.
    std::shared_ptr<REX::tea::procReweightor> build();

private:
    const madspace::MatrixElementApi* _normaliser = nullptr;
    std::vector<UmamiInputKey> _normaliser_input_keys;
    std::vector<UmamiOutputKey> _normaliser_output_keys;

    std::vector<const madspace::MatrixElementApi*> _apis;
    std::vector<std::vector<UmamiInputKey>> _input_keys_list;
    std::vector<std::vector<UmamiOutputKey>> _output_keys_list;
    std::vector<UmamiOutputKey> _result_keys;

    std::vector<std::shared_ptr<REX::tea::weightor>> _weightors;
    std::shared_ptr<REX::eventBelongs> _event_checker = nullptr;
    REX::event_bool_fn _event_checker_fn = nullptr;
};

// WareHouse aggregates one or more per-process TupperWare specifications together
// with the reweightor-level plumbing (initialise/finalise/iterators/launch_names)
// into a single REX::tea::reweightor, which it owns. Each registered TupperWare
// becomes one entry in the resulting reweightor::reweightors (ie one
// procReweightor per physical process, matching the "one TupperWare per process"
// model documented on TupperWare above); the REX::lhe passed to build() supplies
// the lhe base that reweightor::extract_sorter() reads events from and sorts into
// those per-process procReweightors via their event checkers.
//
// TupperWares are retained after build(), so the same registered apis/checkers can
// be rebuilt against a different REX::lhe by calling build() again.
class WareHouse {
public:
    WareHouse() = default;
    WareHouse(const WareHouse&) = delete;
    WareHouse& operator=(const WareHouse&) = delete;
    WareHouse(WareHouse&&) noexcept = default;
    WareHouse& operator=(WareHouse&&) noexcept = default;

    // Convenience: appends a new, empty TupperWare and returns a reference to it
    // for in-place configuration, eg house.add_process().add_api(api).set_event_checker(chk).
    TupperWare& add_process();

    // Reweighting-loop plumbing, mirroring REX::tea::reweightor's own setters:
    // per-iteration hooks (eg SLHA card rewriting between amplitude evaluations),
    // hooks run once before/after the iteration loop, and the weight-id names
    // recorded for each iteration's appended weights. Optional: if never called,
    // build() leaves the corresponding reweightor default in place (a single
    // no-op iteration, true_function initialise/finalise, no launch_names).
    WareHouse& set_iterators(std::vector<REX::tea::iterator> iters);
    WareHouse& add_iterator(REX::tea::iterator iter);
    WareHouse& set_initialise(REX::tea::iterator init);
    WareHouse& set_finalise(REX::tea::iterator fin);
    WareHouse& set_launch_names(std::vector<std::string> names);
    WareHouse& add_launch_name(const std::string& name);

    // Builds (or rebuilds) the owned reweightor from the registered TupperWares'
    // procReweightors and the given REX::lhe, which becomes the reweightor's lhe
    // base. Throws if no processes have been registered. Repeated calls discard
    // any previously-built reweightor.
    REX::tea::reweightor& build(REX::lhe&& mother);
    REX::tea::reweightor& build(const REX::lhe& mother);

    // Access to the owned reweightor. Throws if build() has not been called yet.
    REX::tea::reweightor& get();
    const REX::tea::reweightor& get() const;
    bool built() const noexcept;

    // Direct access to TupperWare; getting, setting, and adding
    TupperWare& process(std::size_t index);
    const TupperWare& process(std::size_t index) const;
    std::vector<TupperWare>& processes();
    const std::vector<TupperWare>& processes() const;
    WareHouse& set_processes(std::vector<TupperWare> processes);
    WareHouse& add_process(TupperWare process);
    WareHouse& add_process(std::vector<TupperWare> processes);

private:
    std::vector<TupperWare> _processes;
    std::vector<REX::tea::iterator> _iterators;
    REX::tea::iterator _initialise = REX::tea::true_function;
    REX::tea::iterator _finalise = REX::tea::true_function;
    std::vector<std::string> _launch_names;

    std::unique_ptr<REX::tea::reweightor> _reweightor;
};

} // namespace madtrex
