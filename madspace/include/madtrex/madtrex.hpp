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
#include "tupperware.hpp"

using json = nlohmann::json;

namespace madtrex {

using namespace madspace;

    std::shared_ptr<REX::lhe> load_lhe_xml(const std::string& path);
    std::shared_ptr<REX::lhe> load_lhe_xml(std::istream& stream);
    std::shared_ptr<REX::lhe> load_lhe(const std::string& path);

    
    

}