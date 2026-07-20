#pragma once

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/tensor.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "madspace/umami.h"

namespace madspace {

class MatrixElementApi {
public:
    MatrixElementApi(MatrixElementApi&&) noexcept = default;
    MatrixElementApi& operator=(MatrixElementApi&&) noexcept = default;
    MatrixElementApi(const MatrixElementApi&) = delete;
    MatrixElementApi& operator=(const MatrixElementApi&) = delete;
    DeviceType device_type() const {
        UmamiDevice dev;
        check_umami_status(_get_meta(UMAMI_META_DEVICE, &dev));
        switch (dev) {
        case UMAMI_DEVICE_CPU:
            return DeviceType::cpu;
        case UMAMI_DEVICE_CUDA:
            return DeviceType::cuda;
        case UMAMI_DEVICE_HIP:
            return DeviceType::hip;
        default:
            throw_error("matrix element device not known");
        }
    }
    std::size_t particle_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_PARTICLE_COUNT, &count));
        return count;
    }
    std::size_t diagram_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_DIAGRAM_COUNT, &count));
        return count;
    }
    std::size_t helicity_count() const {
        int count;
        check_umami_status(_get_meta(UMAMI_META_HELICITY_COUNT, &count));
        return count;
    }
    std::size_t index() const { return _index; }
    const std::string& file_name() const { return _file_name; }
    std::vector<bool> supported_inputs() const {
        bool const* data; int count;
        check_umami_status(_supported_inputs(&data, &count));
        std::vector<bool> result(UMAMI_INPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i)
            result[i] = data[i];
        return result;
    }
    std::vector<bool> required_inputs() const {
        bool const* data; int count;
        check_umami_status(_required_inputs(&data, &count));
        std::vector<bool> supported = supported_inputs();
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i) {
            if (data[i] && !supported[i]) {
                throw_error(std::format(
                    "input key {} is reported as required but not as supported", i
                ));
            }
        }
        std::vector<bool> result(UMAMI_INPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i)
            result[i] = data[i];
        return result;
    }
    std::vector<bool> supported_outputs() const {
        bool const* data; int count;
        check_umami_status(_supported_outputs(&data, &count));
        std::vector<bool> result(UMAMI_OUTPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_OUTPUT_KEY_COUNT; ++i)
            result[i] = data[i];
        return result;
    }

    void call(
        UmamiHandle handle,
        size_t count,
        size_t stride,
        size_t offset,
        size_t input_count,
        UmamiInputKey const* input_keys,
        void const* const* inputs,
        size_t output_count,
        UmamiOutputKey const* output_keys,
        void* const* outputs
    ) const {
        check_umami_status(_matrix_element(
            handle,
            count,
            stride,
            offset,
            input_count,
            input_keys,
            inputs,
            output_count,
            output_keys,
            outputs
        ));
    }

    void* process_instance() const { return _instances.get().get(); }

    // Whether the loaded library actually implements umami_set_parameter/
    // umami_get_parameter (both optional symbols; a library built without
    // them still loads fine, it just can't be parameter-reweighted in place
    // and set_parameter()/get_parameter() will throw UMAMI_ERROR_NOT_IMPLEMENTED
    // if called anyway).
    bool supports_set_parameter() const noexcept { return _has_set_parameter; }
    bool supports_get_parameter() const noexcept { return _has_get_parameter; }

    // Directly sets/reads an independent, SLHA-card-level model parameter on
    // this api's process_instance() -- see umami_set_parameter/
    // umami_get_parameter in umami.h for the accepted forms of name (model
    // name, eg "mdl_MT", or SLHA location, eg "mass 6"). Unlike rewriting the
    // param card and reloading the library, this mutates the already-loaded
    // instance in place, so callers evaluating the matrix element
    // concurrently on the same instance must synchronize against that (see
    // madtrex::detail::api_call_mutex, used by both
    // madtrex::TupperWare-derived weightors and madtrex::ParamHandler for
    // exactly this reason). Throws on UMAMI_ERROR_UNKNOWN_PARAMETER (name is
    // not a settable/known parameter) or if the symbol isn't implemented.
    void set_parameter(const std::string& name, double real, double imag = 0.0) const;
    std::pair<double, double> get_parameter(const std::string& name) const;

private:
    MatrixElementApi(
        const std::string& file,
        const std::string& param_card,
        ThreadPool& thread_pool,
        DevicePtr device,
        std::size_t index = 0
    );

    void check_umami_status(UmamiStatus status) const;
    [[noreturn]] void throw_error(const std::string& message) const;
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    decltype(&umami_get_meta) _get_meta;
    decltype(&umami_supported_inputs) _supported_inputs;
    decltype(&umami_required_inputs) _required_inputs;
    decltype(&umami_supported_outputs) _supported_outputs;
    decltype(&umami_initialize) _initialize;
    decltype(&umami_matrix_element) _matrix_element;
    decltype(&umami_set_parameter) _set_parameter;
    decltype(&umami_get_parameter) _get_parameter;
    bool _has_set_parameter;
    bool _has_get_parameter;
    decltype(&umami_free) _free;
    using InstanceType = std::unique_ptr<void, std::function<void(void*)>>;
    ThreadResource<InstanceType> _instances;
    std::string _file_name;
    std::size_t _index;

    friend class Context;
};

class Context {
    /**
     * Contains global variables and matrix elements
     */
public:
    Context(int thread_count = -1) :
        _device(cpu_device()),
        _thread_pool(std::make_unique<ThreadPool>(thread_count)) {}
    Context(DevicePtr device, int thread_count = -1) :
        _device(device), _thread_pool(std::make_unique<ThreadPool>(thread_count)) {}
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    const MatrixElementApi&
    load_matrix_element(const std::string& file, const std::string& param_card);
    Tensor define_global(
        const std::string& name,
        DataType dtype,
        const SizeVec& shape,
        bool requires_grad = false
    );
    Tensor global(const std::string& name);
    bool global_requires_grad(const std::string& name);
    void set_global_requires_grad(const std::string& name, bool value);
    bool global_exists(const std::string& name);
    std::vector<std::string> global_names() const;
    void delete_global(const std::string& name);
    void copy_globals_from(Context& context);
    Tensor reallocate_globals_contiguously(const std::vector<std::string>& names);
    const MatrixElementApi& matrix_element(std::size_t index) const;
    void save_globals(const std::string& dir) const;
    void load_globals(const std::string& dir);
    DevicePtr device() { return _device; }
    ThreadPool& thread_pool() { return *_thread_pool; }

private:
    DevicePtr _device;
    std::unique_ptr<ThreadPool> _thread_pool;
    std::unordered_map<std::string, std::pair<Tensor, bool>> _globals;
    std::vector<std::unique_ptr<MatrixElementApi>> _matrix_elements;
    std::vector<std::string> _param_card_paths;
};

using ContextPtr = std::shared_ptr<Context>;

ContextPtr default_context();
ContextPtr default_cuda_context(std::size_t index = 0);
ContextPtr default_hip_context(std::size_t index = 0);
ContextPtr default_device_context(DevicePtr device);

inline std::string prefixed_name(const std::string& prefix, const std::string& name) {
    return prefix == "" ? name : std::format("{}.{}", prefix, name);
}

} // namespace madspace
