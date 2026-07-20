#include "madtrex/madtrex.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace madtrex {

namespace {

std::vector<long int> read_pdg_array(const json& arr) {
    std::vector<long int> result;
    result.reserve(arr.size());
    for (auto& v : arr) result.push_back(v.get<long int>());
    return result;
}

std::vector<FlavorChannel> expand_flavor_entry(const json& entry) {
    int index = entry.at("index").get<int>();
    bool mirror = entry.value("mirror", false);

    std::vector<FlavorChannel> result;
    for (auto& option : entry.at("options")) {
        auto pdgs = read_pdg_array(option);
        result.push_back(FlavorChannel{index, pdgs, false});
        if (mirror) {
            if (pdgs.size() < 2) {
                throw std::runtime_error(
                    "parse_subprocesses_json: flavor option has mirror=true but fewer "
                    "than 2 external legs"
                );
            }
            auto swapped = pdgs;
            std::swap(swapped[0], swapped[1]);
            result.push_back(FlavorChannel{index, std::move(swapped), true});
        }
    }
    return result;
}

// Concatenates an event's incoming-leg PDGs (status == -1, in event storage
// order) followed by its outgoing-leg PDGs (status == +1, in event storage
// order). Legs with any other status (eg intermediate resonances) are not
// part of this tuple.
std::vector<long int> ordered_pdgs(const REX::event& e) {
    std::vector<long int> incoming, outgoing;
    for (std::size_t i = 0; i < e.size(); ++i) {
        if (e.status_[i] == -1) incoming.push_back(e.pdg_[i]);
        else if (e.status_[i] == 1) outgoing.push_back(e.pdg_[i]);
    }
    incoming.insert(incoming.end(), outgoing.begin(), outgoing.end());
    return incoming;
}

std::vector<double> ordered_spins(const REX::event& e) {
    std::vector<double> incoming, outgoing;
    for (std::size_t i = 0; i < e.size(); ++i) {
        if (e.status_[i] == -1) incoming.push_back(e.spin_[i]);
        else if (e.status_[i] == 1) outgoing.push_back(e.spin_[i]);
    }
    incoming.insert(incoming.end(), outgoing.begin(), outgoing.end());
    return incoming;
}

const FlavorChannel* match_flavor(const SubProcessSpec& spec, const REX::event& e) {
    auto pdgs = ordered_pdgs(e);
    for (auto& fc : spec.flavors) {
        if (fc.pdgs == pdgs) return &fc;
    }
    return nullptr;
}

bool is_fixed_helicity(double spin) {
    return std::abs(spin - 1.0) < 1e-6 || std::abs(spin + 1.0) < 1e-6;
}

// The UMAMI libraries MadMatrix actually generates don't implement the
// *optional* introspection functions (umami_supported_inputs/
// required_inputs/supported_outputs), which is what make_weightor's
// auto-derive path (the single-argument TupperWare::add_api overload) needs
// to work at all -- against these libraries it just throws. So Driver
// always builds its own explicit key list instead of relying on
// auto-derivation: momenta and alpha_s are requested unconditionally (every
// subprocess needs the former, and QCD amplitudes need the latter), and the
// flavor index only when the subprocess actually distinguishes more than one
// flavor channel (requesting it unconditionally would mean depending on
// UMAMI_IN_FLAVOR_INDEX support that a single-channel subprocess's library
// may not have). The helicity index is intentionally never requested here:
// this library hard-errors on UMAMI_IN_HELICITY_INDEX, and its behavior when
// no helicity input is given at all appears to be a single fixed-random-draw
// evaluation (UMAMI_IN_RANDOM_HELICITY defaults to 0.5 internally) rather
// than a true sum over helicities -- a real physics-affecting subtlety in how
// this specific UMAMI implementation handles helicities that's worth
// resolving deliberately later, not papered over here. SubProcessSpec's own
// helicity_index (via make_event_checker) still runs regardless, so
// event::helicity_ is populated whenever an event carries genuine per-leg
// helicities, ready for whichever explicit-helicity handling gets added.
std::vector<UmamiInputKey> input_keys_for(const SubProcessSpec& spec) {
    std::vector<UmamiInputKey> keys{UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S};
    std::set<int> distinct_indices;
    for (auto& fc : spec.flavors) distinct_indices.insert(fc.index);
    if (distinct_indices.size() > 1) keys.push_back(UMAMI_IN_FLAVOR_INDEX);
    return keys;
}

} // namespace

std::vector<SubProcessSpec> parse_subprocesses_json(const std::string& subprocesses_json_path) {
    std::ifstream in(subprocesses_json_path);
    if (!in) {
        throw std::runtime_error(
            "parse_subprocesses_json: failed to open " + subprocesses_json_path
        );
    }
    json data;
    in >> data;
    if (!data.is_array()) {
        throw std::runtime_error(
            "parse_subprocesses_json: expected a top-level JSON array in " +
            subprocesses_json_path
        );
    }

    std::vector<SubProcessSpec> specs;
    specs.reserve(data.size());
    for (auto& entry : data) {
        SubProcessSpec spec;
        spec.incoming = read_pdg_array(entry.at("incoming"));
        spec.outgoing = read_pdg_array(entry.at("outgoing"));
        spec.me_path_template = entry.at("me_path").get<std::string>();
        spec.source_path = entry.value("path", std::string());
        spec.diagram_count = entry.value("diagram_count", std::size_t(0));

        for (auto& flavor_entry : entry.at("flavors")) {
            auto expanded = expand_flavor_entry(flavor_entry);
            spec.flavors.insert(spec.flavors.end(), expanded.begin(), expanded.end());
        }

        if (entry.contains("helicities")) {
            for (auto& hel : entry.at("helicities")) {
                spec.helicities.push_back(hel.get<std::vector<int>>());
            }
        }

        specs.push_back(std::move(spec));
    }
    return specs;
}

REX::eventBelongs make_event_checker(const SubProcessSpec& spec) {
    REX::eventBelongs checker;

    // One template event per un-mirrored flavor option: the default
    // external_legs_comparator matches purely on PDG multiset per status,
    // which mirroring doesn't change, so mirrored variants don't need their
    // own template.
    for (auto& fc : spec.flavors) {
        if (fc.mirrored) continue;
        std::size_t n = fc.pdgs.size();
        REX::event templ(n);
        std::vector<short int> status(n, 1);
        for (std::size_t i = 0; i < spec.incoming.size() && i < n; ++i) status[i] = -1;
        templ.set_pdg(fc.pdgs);
        templ.set_status(status);
        checker.add_event(templ);
    }

    checker.flavor_index = [spec](REX::event& e) -> std::size_t {
        auto* fc = match_flavor(spec, e);
        if (!fc) {
            throw std::runtime_error(
                "make_event_checker: event matched the subprocess by PDG content but "
                "not positionally by any known flavor channel; the assumed event leg "
                "ordering (incoming, then outgoing, both in event storage order) may "
                "not hold for this event"
            );
        }
        return static_cast<std::size_t>(fc->index);
    };

    checker.helicity_index = [spec](REX::event& e) -> std::size_t {
        auto* fc = match_flavor(spec, e);
        if (!fc) {
            throw std::runtime_error(
                "make_event_checker: event matched the subprocess by PDG content but "
                "not positionally by any known flavor channel"
            );
        }
        auto spins = ordered_spins(e);
        if (fc->mirrored && spins.size() >= 2) std::swap(spins[0], spins[1]);

        for (double s : spins) {
            if (!is_fixed_helicity(s)) return REX::npos; // no fixed helicity: sum over helicities
        }
        std::vector<int> rounded;
        rounded.reserve(spins.size());
        for (double s : spins) rounded.push_back(s > 0 ? 1 : -1);

        for (std::size_t i = 0; i < spec.helicities.size(); ++i) {
            if (spec.helicities[i] == rounded) return i;
        }
        throw std::runtime_error(
            "make_event_checker: event has fixed (+-1) helicities that don't match any "
            "entry in the subprocess's helicities list"
        );
    };

    return checker;
}

std::string resolve_me_path(
    const std::string& process_directory,
    const std::string& me_path_template,
    const std::vector<std::string>& device_priority
) {
    namespace fs = std::filesystem;
    fs::path full_template = fs::path(process_directory) / me_path_template;
    std::string filename_template = full_template.filename().string();
    fs::path dir = full_template.parent_path();

    static const std::string placeholder = "{device}";
    auto pos = filename_template.find(placeholder);
    if (pos == std::string::npos) {
        if (!fs::exists(full_template)) {
            throw std::runtime_error(
                "resolve_me_path: " + full_template.string() + " does not exist"
            );
        }
        return full_template.string();
    }

    std::string prefix = filename_template.substr(0, pos);
    std::string suffix = filename_template.substr(pos + placeholder.size());

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error(
            "resolve_me_path: directory " + dir.string() + " does not exist"
        );
    }

    std::vector<std::string> found;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        found.push_back(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size()));
    }

    if (found.empty()) {
        throw std::runtime_error(
            "resolve_me_path: no library matching " + prefix + "*" + suffix + " found in " +
            dir.string()
        );
    }

    for (auto& preferred : device_priority) {
        if (std::find(found.begin(), found.end(), preferred) != found.end()) {
            return (dir / (prefix + preferred + suffix)).string();
        }
    }

    if (found.size() == 1) {
        return (dir / (prefix + found.front() + suffix)).string();
    }

    std::string found_list;
    for (auto& f : found) {
        if (!found_list.empty()) found_list += ", ";
        found_list += f;
    }
    throw std::runtime_error(
        "resolve_me_path: multiple device variants found for " + prefix + "*" + suffix +
        " in " + dir.string() + " (" + found_list + ") and none is listed in the device "
        "priority; call Driver::set_device_priority(...) to disambiguate"
    );
}

Driver::Driver() : _context(madspace::default_context()) {}

Driver::Driver(madspace::ContextPtr context) : _context(std::move(context)) {}

Driver& Driver::set_device_priority(std::vector<std::string> priority) {
    _device_priority = std::move(priority);
    return *this;
}

const std::vector<std::string>& Driver::device_priority() const { return _device_priority; }

Driver& Driver::load_process(
    const std::string& subprocesses_json_path, const std::string& param_card
) {
    namespace fs = std::filesystem;
    fs::path json_path(subprocesses_json_path);
    fs::path process_directory = json_path.parent_path().parent_path();

    std::string resolved_param_card = param_card;
    if (resolved_param_card.empty()) {
        resolved_param_card = (process_directory / "Cards" / "param_card.dat").string();
    }

    auto specs = parse_subprocesses_json(subprocesses_json_path);
    for (auto& spec : specs) {
        std::string me_path =
            resolve_me_path(process_directory.string(), spec.me_path_template, _device_priority);
        const auto& api = _context->load_matrix_element(me_path, resolved_param_card);
        auto checker = make_event_checker(spec);
        _warehouse.add_process()
            .add_api(api, input_keys_for(spec), {UMAMI_OUT_MATRIX_ELEMENT})
            .set_event_checker(std::move(checker));
        _specs.push_back(std::move(spec));
    }
    return *this;
}

Driver& Driver::load_process_directory(
    const std::string& process_directory, const std::string& param_card
) {
    namespace fs = std::filesystem;
    fs::path json_path = fs::path(process_directory) / "SubProcesses" / "subprocesses.json";
    return load_process(json_path.string(), param_card);
}

madspace::ContextPtr Driver::context() const { return _context; }

WareHouse& Driver::warehouse() { return _warehouse; }

const WareHouse& Driver::warehouse() const { return _warehouse; }

const std::vector<SubProcessSpec>& Driver::specs() const { return _specs; }

} // namespace madtrex
