#include "madtrex/tupperware.hpp"

#include <algorithm>
#include <format>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace madtrex {

namespace detail {

std::mutex& api_call_mutex(const madspace::MatrixElementApi& api) {
    static std::mutex registry_mutex;
    static std::unordered_map<const madspace::MatrixElementApi*, std::unique_ptr<std::mutex>>
        registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto& slot = registry[&api];
    if (!slot) {
        slot = std::make_unique<std::mutex>();
    }
    return *slot;
}

} // namespace detail

namespace {

using madspace::MatrixElementApi;

// umami_supported_inputs/umami_required_inputs/umami_supported_outputs are optional
// symbols: MatrixElementApi falls back to a stub that reports
// UMAMI_ERROR_NOT_IMPLEMENTED, which the query methods turn into a thrown
// runtime_error. Treat "not implemented" as "unknown" rather than propagating.
std::optional<std::vector<bool>> try_supported_inputs(const MatrixElementApi& api) {
    try {
        return api.supported_inputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<std::vector<bool>> try_required_inputs(const MatrixElementApi& api) {
    try {
        return api.required_inputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<std::vector<bool>> try_supported_outputs(const MatrixElementApi& api) {
    try {
        return api.supported_outputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// LHECompleter (madspace/src/driver/lhe_output.cpp, complete_event_data())
// can insert status==2 intermediate resonance lines into events whose
// propagator falls inside its Breit-Wigner window. Those lines matter for
// LHE consumers but are not part of what a compiled matrix element expects
// as external legs (api.particle_count() only counts incoming+outgoing
// partons), so the per-event view fed to UMAMI must exclude them. Builds a
// fresh process from copies of proc.events with per-event indices
// restricted to non-intermediate (status != 2) partons, leaving proc and
// its underlying (possibly shared) events untouched.
REX::process external_legs_process(REX::process& proc) {
    std::vector<REX::event> filtered_events;
    filtered_events.reserve(proc.events.size());
    for (auto& ev_ptr : proc.events) {
        REX::event copy = *ev_ptr;
        std::vector<std::size_t> external_indices;
        external_indices.reserve(copy.n_);
        for (std::size_t i = 0; i < copy.n_; ++i) {
            if (copy.status_[i] != 2) external_indices.push_back(i);
        }
        copy.set_indices(external_indices);
        filtered_events.push_back(std::move(copy));
    }
    return REX::process(std::move(filtered_events), /*filter_partons=*/true, /*column_major=*/false);
}

void check_particle_counts(const MatrixElementApi& api, REX::process& proc) {
    std::size_t expected = api.particle_count();
    for (std::size_t i = 0; i < proc.size(); ++i) {
        std::size_t begin = (i == 0) ? 0 : proc.n_summed[i - 1];
        std::size_t end = proc.n_summed[i];
        if (end - begin != expected) {
            throw std::runtime_error(std::format(
                "make_weightor: event {} has {} particles, but matrix element {} "
                "expects {}",
                i,
                end - begin,
                api.file_name(),
                expected
            ));
        }
    }
}

// Check whether the process has been transposed to the UMAMI
// memory layout, and if not, transpose it
// Also checks that the number of particles in the process
// matches the number expected by the matrix element
bool gather_momenta(const MatrixElementApi& api, REX::process& proc) {
    std::size_t count = proc.size();
    std::size_t particles = api.particle_count();
    if (proc.umami_momenta().size() == 0) proc.to_umami();
    if (proc.umami_momenta().size() != count * particles * 4) {
        throw std::runtime_error(std::format(
            "make_weightor: process has {} events, but matrix element {} expects "
            "{} particles per event",
            count,
            api.file_name(),
            particles
        ));
    }
    return true;
}

std::shared_ptr<std::vector<double>> evaluate(
    const MatrixElementApi& api,
    REX::process& proc,
    const std::vector<UmamiInputKey>& input_keys,
    const std::vector<UmamiOutputKey>& output_keys,
    const std::vector<UmamiOutputKey>& result_keys
) {
    std::size_t count = proc.size();
    if (count == 0) {
        return std::make_shared<std::vector<double>>();
    }
    // external_proc drops any status==2 intermediate resonance lines LHECompleter
    // may have inserted, so particle counts/momenta match what the matrix element
    // expects (its external legs); event-level quantities (alphaS/flavor/helicity
    // below) are unaffected by that filtering and still come from proc directly.
    REX::process external_proc = external_legs_process(proc);
    check_particle_counts(api, external_proc);

    // Kept alive until after the call below, since input_ptrs points into these.
    std::vector<double> momenta_buf;

    std::vector<UmamiInputKey> active_inputs;
    std::vector<void const*> input_ptrs;
    active_inputs.reserve(input_keys.size());
    input_ptrs.reserve(input_keys.size());

    for (auto key : input_keys) {
        switch (key) {
        case UMAMI_IN_MOMENTA:
            if (!gather_momenta(api, external_proc)) {
                throw std::runtime_error(
                    "make_weightor: failed to gather momenta for UMAMI"
                );
            }
            active_inputs.push_back(key);
            input_ptrs.push_back(external_proc.umami_momenta().data());
            break;
        case UMAMI_IN_ALPHA_S:
            if (proc.alphaS().size() != count) {
                throw std::runtime_error(
                    "make_weightor: process does not carry a per-event alpha_s "
                    "value for every event"
                );
            }
            active_inputs.push_back(key);
            input_ptrs.push_back(proc.alphaS().data());
            break;
        case UMAMI_IN_FLAVOR_INDEX:
            if (proc.flavor().size() != count) {
                throw std::runtime_error(
                    "make_weightor: process does not carry a per-event flavor "
                    "index for every event"
                );
            }
            active_inputs.push_back(key);
            input_ptrs.push_back(proc.flavor().data());
            break;
        case UMAMI_IN_HELICITY_INDEX:
            if (proc.helicity().size() != count) {
                throw std::runtime_error(
                    "make_weightor: process does not carry a per-event helicity "
                    "index for every event"
                );
            }
            active_inputs.push_back(key);
            input_ptrs.push_back(proc.helicity().data());
            break;
        default:
            throw std::runtime_error(
                "make_weightor: input key cannot be derived from a REX::process"
            );
        }
    }

    std::unordered_map<UmamiOutputKey, std::vector<double>> output_buffers;
    std::vector<void*> output_ptrs;
    output_ptrs.reserve(output_keys.size());
    for (auto key : output_keys) {
        std::vector<double> buf;
        switch (key) {
        case UMAMI_OUT_MATRIX_ELEMENT:
            buf.assign(count, 0.0);
            break;
        case UMAMI_OUT_DIAGRAM_AMP2:
            buf.assign(api.diagram_count() * count, 0.0);
            break;
        default:
            throw std::runtime_error(
                "make_weightor: output key is not a reweighting-usable value"
            );
        }
        auto [it, inserted] = output_buffers.emplace(key, std::move(buf));
        output_ptrs.push_back(it->second.data());
    }

    {
        std::lock_guard<std::mutex> guard(detail::api_call_mutex(api));
        api.call(
            api.process_instance(),
            count,
            count,
            0,
            active_inputs.size(),
            active_inputs.data(),
            input_ptrs.data(),
            output_keys.size(),
            output_keys.data(),
            output_ptrs.data()
        );
    }

    auto result = std::make_shared<std::vector<double>>(count, 0.0);
    for (auto key : result_keys) {
        auto found = output_buffers.find(key);
        if (found == output_buffers.end()) {
            throw std::runtime_error(
                "make_weightor: result key was not among the requested output keys"
            );
        }
        if (key == UMAMI_OUT_MATRIX_ELEMENT) {
            for (std::size_t e = 0; e < count; ++e) {
                (*result)[e] += found->second[e];
            }
        } else { // UMAMI_OUT_DIAGRAM_AMP2
            std::size_t diagrams = api.diagram_count();
            for (std::size_t e = 0; e < count; ++e) {
                for (std::size_t d = 0; d < diagrams; ++d) {
                    (*result)[e] += found->second[d * count + e];
                }
            }
        }
    }
    return result;
}

} // namespace

std::vector<UmamiInputKey> default_input_keys(const MatrixElementApi& api) {
    auto supported = try_supported_inputs(api);
    if (!supported) {
        throw std::runtime_error(std::format(
            "make_weightor: matrix element {} does not report supported inputs; "
            "pass input keys explicitly",
            api.file_name()
        ));
    }
    auto required = try_required_inputs(api);

    static constexpr UmamiInputKey derivable[] = {
        UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_HELICITY_INDEX
    };

    std::vector<UmamiInputKey> keys;
    for (std::size_t i = 0; i < UMAMI_INPUT_KEY_COUNT; ++i) {
        auto key = static_cast<UmamiInputKey>(i);
        bool is_derivable = std::find(std::begin(derivable), std::end(derivable), key) !=
                             std::end(derivable);
        bool is_required = required && (*required)[i];
        if (is_required && !is_derivable) {
            throw std::runtime_error(std::format(
                "make_weightor: matrix element {} requires input key {}, which "
                "cannot be derived from a REX::process; pass input keys explicitly",
                api.file_name(),
                i
            ));
        }
        if (is_derivable && (*supported)[i]) {
            keys.push_back(key);
        }
    }
    return keys;
}

std::vector<UmamiOutputKey> default_output_keys(const MatrixElementApi& api) {
    auto supported = try_supported_outputs(api);
    if (!supported || !(*supported)[UMAMI_OUT_MATRIX_ELEMENT]) {
        throw std::runtime_error(std::format(
            "make_weightor: matrix element {} does not report supporting "
            "UMAMI_OUT_MATRIX_ELEMENT; pass output/result keys explicitly",
            api.file_name()
        ));
    }
    return {UMAMI_OUT_MATRIX_ELEMENT};
}

std::shared_ptr<REX::tea::weightor> make_weightor(const MatrixElementApi& api) {
    return make_weightor(
        api, default_input_keys(api), default_output_keys(api), {UMAMI_OUT_MATRIX_ELEMENT}
    );
}

std::shared_ptr<REX::tea::weightor> make_weightor(
    const MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys
) {
    if (result_keys.empty()) {
        result_keys = {UMAMI_OUT_MATRIX_ELEMENT};
    }
    for (auto key : result_keys) {
        if (std::find(output_keys.begin(), output_keys.end(), key) == output_keys.end()) {
            throw std::runtime_error(
                "make_weightor: result key must be among the requested output keys"
            );
        }
    }

    const MatrixElementApi* api_ptr = &api;
    return std::make_shared<REX::tea::weightor>(
        [api_ptr,
         input_keys = std::move(input_keys),
         output_keys = std::move(output_keys),
         result_keys = std::move(result_keys)](REX::process& proc) {
            return evaluate(*api_ptr, proc, input_keys, output_keys, result_keys);
        }
    );
}

TupperWare::TupperWare(const madspace::MatrixElementApi& api, REX::eventBelongs checker) {
    add_api(api);
    set_normaliser(api);
    set_event_checker(std::move(checker));
}

TupperWare::TupperWare(const madspace::MatrixElementApi& api, REX::event_bool_fn checker) {
    add_api(api);
    set_normaliser(api);
    set_event_checker(std::move(checker));
}

TupperWare& TupperWare::add_api(const madspace::MatrixElementApi& api) {
    return add_api(api, {}, {});
}

TupperWare& TupperWare::add_api(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys
) {
    _apis.push_back(&api);
    _input_keys_list.push_back(std::move(input_keys));
    _output_keys_list.push_back(std::move(output_keys));
    return *this;
}

TupperWare& TupperWare::add_api(std::vector<const madspace::MatrixElementApi*> apis) {
    for (const auto* api : apis) {
        add_api(*api);
    }
    return *this;
}

TupperWare& TupperWare::add_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<UmamiInputKey> input_keys_list,
    std::vector<UmamiOutputKey> output_keys_list
) {
    for (std::size_t i = 0; i < apis.size(); ++i) {
        add_api(*apis[i], input_keys_list, output_keys_list);
    }
    return *this;
}

TupperWare& TupperWare::add_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<std::vector<UmamiInputKey>> input_keys_list,
    std::vector<std::vector<UmamiOutputKey>> output_keys_list
) {
    for (std::size_t i = 0; i < apis.size(); ++i) {
        std::vector<UmamiInputKey> input_keys;
        if (i < input_keys_list.size()) {
            input_keys = std::move(input_keys_list[i]);
        }
        std::vector<UmamiOutputKey> output_keys;
        if (i < output_keys_list.size()) {
            output_keys = std::move(output_keys_list[i]);
        }
        add_api(*apis[i], input_keys, output_keys);
    }
    return *this;
}

TupperWare& TupperWare::set_api(std::vector<const madspace::MatrixElementApi*> apis) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    return add_api(std::move(apis));
}

TupperWare& TupperWare::set_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<UmamiInputKey> input_keys_list,
    std::vector<UmamiOutputKey> output_keys_list
) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    return add_api(std::move(apis), std::move(input_keys_list), std::move(output_keys_list));
}

TupperWare& TupperWare::set_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<std::vector<UmamiInputKey>> input_keys_list,
    std::vector<std::vector<UmamiOutputKey>> output_keys_list
) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    return add_api(std::move(apis), std::move(input_keys_list), std::move(output_keys_list));
}

TupperWare& TupperWare::set_normaliser(const madspace::MatrixElementApi& api) {
    return set_normaliser(api, {}, {});
}

TupperWare& TupperWare::set_normaliser(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys
) {
    _normaliser = &api;
    _normaliser_input_keys = std::move(input_keys);
    _normaliser_output_keys = std::move(output_keys);
    return *this;
}

TupperWare& TupperWare::set_result_keys(std::vector<UmamiOutputKey> keys) {
    _result_keys = std::move(keys);
    return *this;
}

TupperWare& TupperWare::set_event_checker(REX::eventBelongs checker) {
    _event_checker = std::make_shared<REX::eventBelongs>(std::move(checker));
    _event_checker_fn = nullptr;
    return *this;
}

TupperWare& TupperWare::set_event_checker(std::shared_ptr<REX::eventBelongs> checker) {
    _event_checker = std::move(checker);
    _event_checker_fn = nullptr;
    return *this;
}

TupperWare& TupperWare::set_event_checker(REX::event_bool_fn checker) {
    _event_checker = nullptr;
    _event_checker_fn = std::move(checker);
    return *this;
}

std::shared_ptr<REX::tea::procReweightor> TupperWare::build() {
    if (_apis.empty()) {
        throw std::runtime_error("TupperWare::build: no matrix elements registered");
    }
    if (!_event_checker && !_event_checker_fn) {
        throw std::runtime_error("TupperWare::build: no event checker set");
    }

    auto result_keys_for = [this]() -> std::vector<UmamiOutputKey> {
        return _result_keys.empty() ? std::vector<UmamiOutputKey>{UMAMI_OUT_MATRIX_ELEMENT}
                                     : _result_keys;
    };

    _weightors.clear();
    _weightors.reserve(_apis.size());
    std::vector<REX::tea::weightor> rwgts;
    rwgts.reserve(_apis.size());
    for (std::size_t i = 0; i < _apis.size(); ++i) {
        const auto& api = *_apis[i];
        auto input_keys =
            _input_keys_list[i].empty() ? default_input_keys(api) : _input_keys_list[i];
        auto output_keys =
            _output_keys_list[i].empty() ? default_output_keys(api) : _output_keys_list[i];
        auto weightor_ptr = make_weightor(api, input_keys, output_keys, result_keys_for());
        _weightors.push_back(weightor_ptr);
        rwgts.push_back(*weightor_ptr);
    }

    auto proc_rwgt = std::make_shared<REX::tea::procReweightor>();
    proc_rwgt->set_reweight_functions(rwgts);

    if (_event_checker) {
        proc_rwgt->event_checker = _event_checker;
        proc_rwgt->event_checker_fn = _event_checker->get_event_bool();
    } else {
        proc_rwgt->set_event_checker(_event_checker_fn);
    }

    if (_normaliser) {
        auto input_keys = _normaliser_input_keys.empty() ? default_input_keys(*_normaliser)
                                                           : _normaliser_input_keys;
        auto output_keys = _normaliser_output_keys.empty() ? default_output_keys(*_normaliser)
                                                             : _normaliser_output_keys;
        auto norm_weightor = make_weightor(*_normaliser, input_keys, output_keys, result_keys_for());
        proc_rwgt->set_normaliser(*norm_weightor);
    }

    return proc_rwgt;
}

TupperWare& WareHouse::add_process() {
    _processes.emplace_back();
    return _processes.back();
}

WareHouse& WareHouse::set_iterators(std::vector<REX::tea::iterator> iters) {
    _iterators = std::move(iters);
    return *this;
}

WareHouse& WareHouse::add_iterator(REX::tea::iterator iter) {
    _iterators.push_back(std::move(iter));
    return *this;
}

WareHouse& WareHouse::set_initialise(REX::tea::iterator init) {
    _initialise = std::move(init);
    return *this;
}

WareHouse& WareHouse::set_finalise(REX::tea::iterator fin) {
    _finalise = std::move(fin);
    return *this;
}

WareHouse& WareHouse::set_launch_names(std::vector<std::string> names) {
    _launch_names = std::move(names);
    return *this;
}

WareHouse& WareHouse::add_launch_name(const std::string& name) {
    _launch_names.push_back(name);
    return *this;
}

namespace {

std::vector<std::shared_ptr<REX::tea::procReweightor>> build_proc_reweightors(
    std::vector<TupperWare>& processes
) {
    if (processes.empty()) {
        throw std::runtime_error("WareHouse::build: no processes registered");
    }
    std::vector<std::shared_ptr<REX::tea::procReweightor>> result;
    result.reserve(processes.size());
    for (auto& process : processes) {
        result.push_back(process.build());
    }
    return result;
}

} // namespace

REX::tea::reweightor& WareHouse::build(REX::lhe&& mother) {
    auto proc_rwgts = build_proc_reweightors(_processes);
    _reweightor =
        std::make_unique<REX::tea::reweightor>(std::move(mother), std::move(proc_rwgts));
    if (!_iterators.empty()) {
        _reweightor->set_iterators(_iterators);
    }
    if (_initialise) {
        _reweightor->set_initialise(_initialise);
    }
    if (_finalise) {
        _reweightor->set_finalise(_finalise);
    }
    if (!_launch_names.empty()) {
        _reweightor->set_launch_names(_launch_names);
    }
    return *_reweightor;
}

REX::tea::reweightor& WareHouse::build(const REX::lhe& mother) {
    return build(REX::lhe(mother));
}

REX::tea::reweightor& WareHouse::get() {
    if (!_reweightor) {
        throw std::runtime_error("WareHouse::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

const REX::tea::reweightor& WareHouse::get() const {
    if (!_reweightor) {
        throw std::runtime_error("WareHouse::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

bool WareHouse::built() const noexcept { return static_cast<bool>(_reweightor); }

TupperWare& WareHouse::process(std::size_t index) {
    if (index >= _processes.size()) {
        throw std::out_of_range("WareHouse::process: index out of range");
    }
    return _processes[index];
}

const TupperWare& WareHouse::process(std::size_t index) const {
    if (index >= _processes.size()) {
        throw std::out_of_range("WareHouse::process: index out of range");
    }
    return _processes[index];
}

std::vector<TupperWare>& WareHouse::processes() {
    return _processes;
}

const std::vector<TupperWare>& WareHouse::processes() const {
    return _processes;
}

WareHouse& WareHouse::set_processes(std::vector<TupperWare> processes) {
    _processes = std::move(processes);
    return *this;
}

WareHouse& WareHouse::add_process(TupperWare process) {
    _processes.push_back(std::move(process));
    return *this;
}

WareHouse& WareHouse::add_process(std::vector<TupperWare> processes) {
    _processes.insert(_processes.end(),
                      std::make_move_iterator(processes.begin()),
                      std::make_move_iterator(processes.end()));
    return *this;
}

} // namespace madtrex
