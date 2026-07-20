#include "madtrex/binary_io.hpp"

#include <cctype>
#include <stdexcept>

namespace madtrex {

namespace {

using madspace::DataLayout;
using madspace::EventBuffer;
using madspace::EventFile;
using madspace::EventRecord;
using madspace::LHEEvent;
using madspace::LHEMeta;
using madspace::LHEParticle;
using madspace::ParticleRecord;

DataLayout lhe_event_layout() {
    return DataLayout(
        EventRecord::layout(EventRecord::f_lhe_event),
        ParticleRecord::layout(ParticleRecord::f_lhe_particle)
    );
}

// Reads every event out of an already-open, load-mode EventFile laid out with
// EventRecord::f_lhe_event | ParticleRecord::f_lhe_particle. Particles with
// status_code == 0 are dropped: real LHE particles never carry status 0
// (incoming is -1, outgoing +1, intermediate resonance 2), so a 0 marks the
// zero-filled padding EventGenerator::combine_to_lhe_npy appends past an
// event's actual particle count, up to the file's fixed particle_count().
std::vector<LHEEvent> read_binary_events(EventFile& file) {
    // Named local: EventBuffer stores its DataLayout by reference, so the
    // layout must outlive every buffer built from it.
    DataLayout layout = lhe_event_layout();
    std::size_t max_particles = file.particle_count();
    EventBuffer buffer(0, max_particles, layout);

    std::vector<LHEEvent> events;
    events.reserve(file.event_count());

    constexpr std::size_t chunk = 4096;
    while (file.read(buffer, chunk)) {
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            auto rec = buffer.event(i);
            LHEEvent evt;
            evt.process_id = rec.lhe_process_id();
            evt.weight = rec.lhe_weight();
            evt.scale = rec.lhe_scale();
            evt.alpha_qed = rec.lhe_alpha_qed();
            evt.alpha_qcd = rec.lhe_alpha_qcd();
            evt.particles.reserve(max_particles);
            for (std::size_t j = 0; j < max_particles; ++j) {
                auto prec = buffer.particle(i, j);
                int status = prec.lhe_status_code();
                if (status == 0) break; // padding: always a trailing run
                LHEParticle part;
                part.pdg_id = prec.lhe_pdg_id();
                part.status_code = status;
                part.mother1 = prec.lhe_mother1();
                part.mother2 = prec.lhe_mother2();
                part.color = prec.lhe_color();
                part.anti_color = prec.lhe_anti_color();
                part.px = prec.lhe_px();
                part.py = prec.lhe_py();
                part.pz = prec.lhe_pz();
                part.energy = prec.lhe_energy();
                part.mass = prec.lhe_mass();
                part.lifetime = prec.lhe_lifetime();
                part.spin = prec.lhe_spin();
                evt.particles.push_back(part);
            }
            events.push_back(std::move(evt));
        }
    }
    return events;
}

REX::parton lhe_particle_to_parton(const LHEParticle& p) {
    REX::arr2<short int> mother{
        static_cast<short int>(p.mother1), static_cast<short int>(p.mother2)
    };
    REX::arr2<short int> icol{
        static_cast<short int>(p.color), static_cast<short int>(p.anti_color)
    };
    return REX::parton(
        REX::arr4<double>{p.energy, p.px, p.py, p.pz},
        p.mass,
        p.lifetime,
        p.spin,
        static_cast<long int>(p.pdg_id),
        static_cast<short int>(p.status_code),
        mother,
        icol
    );
}

std::shared_ptr<REX::event> lhe_event_to_event(const LHEEvent& raw) {
    std::vector<REX::parton> partons;
    partons.reserve(raw.particles.size());
    for (auto& p : raw.particles) partons.push_back(lhe_particle_to_parton(p));

    auto evt = std::make_shared<REX::event>(std::move(partons));
    evt->set_proc_id(raw.process_id);
    evt->set_weight(raw.weight);
    evt->set_scale(raw.scale);
    evt->set_alphaEW(raw.alpha_qed);
    evt->set_alphaS(raw.alpha_qcd);
    return evt;
}

REX::initNode lhe_meta_to_init(const LHEMeta& meta) {
    REX::initNode init(meta.processes.size());
    init.set_idBm(
        static_cast<long int>(meta.beam1_pdg_id), static_cast<long int>(meta.beam2_pdg_id)
    );
    init.set_eBm(meta.beam1_energy, meta.beam2_energy);
    init.set_pdfG(
        static_cast<short int>(meta.beam1_pdf_authors),
        static_cast<short int>(meta.beam2_pdf_authors)
    );
    init.set_pdfS(
        static_cast<long int>(meta.beam1_pdf_id), static_cast<long int>(meta.beam2_pdf_id)
    );
    for (auto& proc : meta.processes) {
        init.add_xSec(proc.cross_section);
        init.add_xSecErr(proc.cross_section_error);
        init.add_xMax(proc.max_weight);
        init.add_lProc(proc.process_id);
    }
    return init;
}

std::string sanitize_token(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        out.push_back(
            (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') ? c : '_'
        );
    }
    return out.empty() ? std::string("weight") : out;
}

} // namespace

std::shared_ptr<REX::lhe> load_lhe_binary(EventFile& file, const LHEMeta& meta) {
    auto raw_events = read_binary_events(file);

    REX::lheReader<LHEMeta, LHEEvent> reader;
    reader.set_init_translator(lhe_meta_to_init);
    reader.set_event_translator(lhe_event_to_event);

    auto result = reader.read(meta, raw_events);
    return std::make_shared<REX::lhe>(std::move(result));
}

std::shared_ptr<REX::lhe> load_lhe_binary(const std::string& path, const LHEMeta& meta) {
    EventFile file(path, lhe_event_layout(), 0, EventFile::load);
    return load_lhe_binary(file, meta);
}

void save_weights_binary(const REX::lhe& lhe, const std::string& path_prefix) {
    if (!lhe.weight_ids || lhe.weight_ids->empty()) {
        throw std::runtime_error("save_weights_binary: lhe has no weight_ids to write");
    }
    for (std::size_t k = 0; k < lhe.weight_ids->size(); ++k) {
        EventBuffer buffer(lhe.events.size(), 0, madspace::weight_file_layout);
        for (std::size_t i = 0; i < lhe.events.size(); ++i) {
            const auto& evt = lhe.events[i];
            if (!evt || evt->wgts_.size() <= k) {
                throw std::runtime_error(
                    "save_weights_binary: event " + std::to_string(i) +
                    " is missing weight '" + (*lhe.weight_ids)[k] + "'"
                );
            }
            buffer.event(i).weight() = evt->wgts_[k];
        }
        std::string file_name =
            path_prefix + "_" + sanitize_token((*lhe.weight_ids)[k]) + ".npy";
        EventFile out(file_name, madspace::weight_file_layout, 0, EventFile::create);
        out.write(buffer);
    }
}

} // namespace madtrex
