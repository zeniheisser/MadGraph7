#include "madtrex.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "madtrex/tupperware.hpp"

namespace py = pybind11;
using namespace madtrex;
using madspace::MatrixElementApi;

void bind_madtrex(py::module_ m) {
    m.doc() = "Python bindings for madtrex::TupperWare/WareHouse, the generic "
               "MatrixElementApi <-> REX::tea reweighting bridge";

    // TupperWare/WareHouse build REX::tea::procReweightor/reweightor objects and
    // consume REX::eventBelongs/event_bool_fn event checkers -- all types already
    // bound by the separately-built `rex` package (rex / rex.tea). Import them
    // first so those types are registered in the process before any madtrex
    // binding below references them, and do NOT re-register py::classh<...> for
    // any of them here: pybind11 raises "generic_type: type ... is already
    // registered!" the moment both `rex` and `madspace` are imported into the
    // same process and both sides independently declare a class binding for the
    // same C++ type. Cross-module reuse (verified empirically: an independently
    // compiled procReweightor, returned from a second, unrelated extension
    // module, casts cleanly to the already-registered rex.tea.ProcReweightor
    // Python type) is what makes objects built here actually usable with
    // rex.tea.Reweightor.run() etc.
    py::module_::import("rex");
    py::module_::import("rex.tea");

    // ---- UmamiInputKey / UmamiOutputKey ----
    // Plain C enums from madspace/umami.h; not bound anywhere else in
    // _madspace_py. Every value is exposed (not just the subset TupperWare can
    // auto-derive from a REX::process) since the enum also indexes
    // MatrixElementApi.supported_inputs()/required_inputs()/supported_outputs().
    py::enum_<UmamiInputKey>(m, "UmamiInputKey")
        .value("MOMENTA", UMAMI_IN_MOMENTA)
        .value("ALPHA_S", UMAMI_IN_ALPHA_S)
        .value("FLAVOR_INDEX", UMAMI_IN_FLAVOR_INDEX)
        .value("RANDOM_COLOR", UMAMI_IN_RANDOM_COLOR)
        .value("RANDOM_HELICITY", UMAMI_IN_RANDOM_HELICITY)
        .value("RANDOM_DIAGRAM", UMAMI_IN_RANDOM_DIAGRAM)
        .value("HELICITY_INDEX", UMAMI_IN_HELICITY_INDEX)
        .value("DIAGRAM_INDEX", UMAMI_IN_DIAGRAM_INDEX)
        .value("CHANNEL_INDEX", UMAMI_IN_CHANNEL_INDEX);

    py::enum_<UmamiOutputKey>(m, "UmamiOutputKey")
        .value("MATRIX_ELEMENT", UMAMI_OUT_MATRIX_ELEMENT)
        .value("DIAGRAM_AMP2", UMAMI_OUT_DIAGRAM_AMP2)
        .value("COLOR_INDEX", UMAMI_OUT_COLOR_INDEX)
        .value("HELICITY_INDEX", UMAMI_OUT_HELICITY_INDEX)
        .value("DIAGRAM_INDEX", UMAMI_OUT_DIAGRAM_INDEX)
        .value("GPU_STREAM", UMAMI_OUT_GPU_STREAM);

    // default_input_keys/default_output_keys are exposed as the documented
    // starting point for a manual add_api/set_normaliser override.
    // make_weightor() itself is intentionally not bound: it returns a raw
    // shared_ptr<REX::tea::weightor>, which -- like the plain weightor type in
    // tearex.cpp -- isn't something any bound API can consume back from Python
    // (procReweightor's Python constructors only take Python-callable
    // std::functions, not a pre-built weightor handle). TupperWare/WareHouse's
    // add_api()/build() cover the actual construction path end to end.
    m.def("default_input_keys", &default_input_keys, py::arg("api"));
    m.def("default_output_keys", &default_output_keys, py::arg("api"));

    // ---- TupperWare ----
    // set_event_checker is bound only in its shared_ptr<eventBelongs> form, not
    // the sibling by-value eventBelongs overload: a Python EventBelongs object
    // converts to shared_ptr<...> transparently via classh either way, and
    // pybind11 can't otherwise distinguish which of the two overloads a given
    // EventBelongs argument was meant for (same rationale as
    // procReweightor's selector argument in Rex's own tearex.cpp). The
    // convenience constructor taking a plain eventBelongs has no shared_ptr
    // sibling to conflict with, so it's bound as-is.
    py::classh<TupperWare>(m, "TupperWare")
        .def(py::init<>())
        .def(
            py::init<const MatrixElementApi&, REX::eventBelongs>(),
            py::arg("api"),
            py::arg("checker")
        )
        .def(
            py::init<const MatrixElementApi&, REX::event_bool_fn>(),
            py::arg("api"),
            py::arg("checker")
        )
        .def(
            "add_api",
            py::overload_cast<const MatrixElementApi&>(&TupperWare::add_api),
            py::arg("api")
        )
        .def(
            "add_api",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::add_api),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys")
        )
        .def(
            "add_api",
            py::overload_cast<std::vector<const MatrixElementApi*>>(&TupperWare::add_api),
            py::arg("apis")
        )
        .def(
            "add_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::add_api),
            py::arg("apis"),
            py::arg("input_keys"),
            py::arg("output_keys")
        )
        .def(
            "add_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<std::vector<UmamiInputKey>>,
                std::vector<std::vector<UmamiOutputKey>>>(&TupperWare::add_api),
            py::arg("apis"),
            py::arg("input_keys_list"),
            py::arg("output_keys_list")
        )
        .def(
            "set_api",
            py::overload_cast<std::vector<const MatrixElementApi*>>(&TupperWare::set_api),
            py::arg("apis")
        )
        .def(
            "set_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::set_api),
            py::arg("apis"),
            py::arg("input_keys"),
            py::arg("output_keys")
        )
        .def(
            "set_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<std::vector<UmamiInputKey>>,
                std::vector<std::vector<UmamiOutputKey>>>(&TupperWare::set_api),
            py::arg("apis"),
            py::arg("input_keys_list"),
            py::arg("output_keys_list")
        )
        .def(
            "set_normaliser",
            py::overload_cast<const MatrixElementApi&>(&TupperWare::set_normaliser),
            py::arg("api")
        )
        .def(
            "set_normaliser",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::set_normaliser),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys")
        )
        .def("set_result_keys", &TupperWare::set_result_keys, py::arg("keys"))
        .def(
            "set_event_checker",
            py::overload_cast<std::shared_ptr<REX::eventBelongs>>(
                &TupperWare::set_event_checker
            ),
            py::arg("checker")
        )
        .def(
            "set_event_checker",
            py::overload_cast<REX::event_bool_fn>(&TupperWare::set_event_checker),
            py::arg("checker")
        )
        .def("build", &TupperWare::build);

    // ---- WareHouse ----
    // Move-only (copy ctor/assignment deleted in C++, mirroring
    // REX::tea::reweightor's own single-owner semantics); py::classh's
    // smart_holder handles that without needing a copy constructor. The
    // lhe&&-taking build() overload is skipped, same rationale as Reweightor's
    // own constructor in tearex.cpp: Python has no rvalue distinction, and the
    // const lhe& overload covers every practical call (it just wraps its
    // argument in a local REX::lhe copy before delegating).
    py::classh<WareHouse>(m, "WareHouse")
        .def(py::init<>())
        .def(
            "add_process",
            py::overload_cast<TupperWare>(&WareHouse::add_process),
            py::arg("process")
        )
        .def(
            "add_process",
            py::overload_cast<std::vector<TupperWare>>(&WareHouse::add_process),
            py::arg("processes")
        )
        .def(
            "add_process",
            py::overload_cast<>(&WareHouse::add_process),
            py::return_value_policy::reference_internal
        )
        .def("set_processes", &WareHouse::set_processes, py::arg("processes"))
        .def(
            "process",
            py::overload_cast<std::size_t>(&WareHouse::process),
            py::arg("index"),
            py::return_value_policy::reference_internal
        )
        .def("processes", py::overload_cast<>(&WareHouse::processes))
        .def("set_iterators", &WareHouse::set_iterators, py::arg("iterators"))
        .def("add_iterator", &WareHouse::add_iterator, py::arg("iterator"))
        .def("set_initialise", &WareHouse::set_initialise, py::arg("init"))
        .def("set_finalise", &WareHouse::set_finalise, py::arg("fin"))
        .def("set_launch_names", &WareHouse::set_launch_names, py::arg("names"))
        .def("add_launch_name", &WareHouse::add_launch_name, py::arg("name"))
        .def(
            "build",
            py::overload_cast<const REX::lhe&>(&WareHouse::build),
            py::arg("mother"),
            py::return_value_policy::reference_internal
        )
        .def(
            "get",
            py::overload_cast<>(&WareHouse::get),
            py::return_value_policy::reference_internal
        )
        .def("built", &WareHouse::built);
}
