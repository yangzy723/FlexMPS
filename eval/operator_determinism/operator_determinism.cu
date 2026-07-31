#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::ostringstream oss__;                                              \
            oss__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "      \
                  << cudaGetErrorString(err__);                                    \
            throw std::runtime_error(oss__.str());                                 \
        }                                                                         \
    } while (0)

struct Options {
    std::string experiment = "both";      // reduction | gemm | both
    std::string dtype = "all";            // fp32 | fp16 | bf16 | all
    int trials = 100;
    int threads = 256;
    int baseline_shape = -1;               // partitions for reduction; split-K for GEMM
    std::vector<int> variants;             // test partition counts / split-K factors
    std::int64_t reduction_size = 1LL << 24;
    int hidden = 4096;
    int vocab = 4096;
    int exp_min = -4;
    int exp_max = 4;
    std::uint64_t seed = 2027;
    std::string csv_path = "operator_determinism.csv";
    std::string allocation_tag = "unspecified";
};

static std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        values.push_back(std::stoi(item));
    }
    if (values.empty()) {
        throw std::invalid_argument("expected a non-empty comma-separated integer list");
    }
    return values;
}

static bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value after ") + name);
            }
            return argv[++i];
        };

        if (arg == "--experiment") opt.experiment = require_value("--experiment");
        else if (arg == "--dtype") opt.dtype = require_value("--dtype");
        else if (arg == "--trials") opt.trials = std::stoi(require_value("--trials"));
        else if (arg == "--threads") opt.threads = std::stoi(require_value("--threads"));
        else if (arg == "--baseline") opt.baseline_shape = std::stoi(require_value("--baseline"));
        else if (arg == "--variants") opt.variants = parse_int_list(require_value("--variants"));
        else if (arg == "--size") opt.reduction_size = std::stoll(require_value("--size"));
        else if (arg == "--hidden") opt.hidden = std::stoi(require_value("--hidden"));
        else if (arg == "--vocab") opt.vocab = std::stoi(require_value("--vocab"));
        else if (arg == "--exp-min") opt.exp_min = std::stoi(require_value("--exp-min"));
        else if (arg == "--exp-max") opt.exp_max = std::stoi(require_value("--exp-max"));
        else if (arg == "--seed") opt.seed = std::stoull(require_value("--seed"));
        else if (arg == "--csv") opt.csv_path = require_value("--csv");
        else if (arg == "--allocation-tag") opt.allocation_tag = require_value("--allocation-tag");
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./operator_determinism [options]\n"
                << "  --experiment reduction|gemm|both\n"
                << "  --dtype fp32|fp16|bf16|all\n"
                << "  --trials N\n"
                << "  --threads N                 power of two, <= 1024\n"
                << "  --baseline N                partitions or split-K factor\n"
                << "  --variants a,b,c            tested partitions or split-K factors\n"
                << "  --size N                    reduction vector length\n"
                << "  --hidden K                  LM-head hidden dimension\n"
                << "  --vocab N                   number of output logits\n"
                << "  --exp-min N                 minimum base-2 input exponent (default: -4)\n"
                << "  --exp-max N                 maximum base-2 input exponent (default: 4)\n"
                << "  --seed N\n"
                << "  --csv path\n"
                << "  --allocation-tag text       metadata only; e.g., pctx-8sm\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!is_power_of_two(opt.threads) || opt.threads > 1024) {
        throw std::invalid_argument("--threads must be a power of two in [1, 1024]");
    }
    if (opt.trials <= 0) throw std::invalid_argument("--trials must be positive");
    if (opt.reduction_size <= 0) throw std::invalid_argument("--size must be positive");
    if (opt.hidden <= 0 || opt.vocab <= 0) {
        throw std::invalid_argument("--hidden and --vocab must be positive");
    }
    if (opt.experiment != "reduction" && opt.experiment != "gemm" && opt.experiment != "both") {
        throw std::invalid_argument("--experiment must be reduction, gemm, or both");
    }
    if (opt.dtype != "fp32" && opt.dtype != "fp16" && opt.dtype != "bf16" && opt.dtype != "all") {
        throw std::invalid_argument("--dtype must be fp32, fp16, bf16, or all");
    }
    if (opt.exp_min > opt.exp_max || opt.exp_min < -14 || opt.exp_max > 14) {
        throw std::invalid_argument(
            "exponent range must satisfy -14 <= --exp-min <= --exp-max <= 14");
    }
    return opt;
}

template <typename T>
__host__ __device__ inline float scalar_to_float(T value);

template <>
__host__ __device__ inline float scalar_to_float<float>(float value) {
    return value;
}

template <>
__host__ __device__ inline float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__host__ __device__ inline float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__host__ __device__ inline T float_to_scalar(float value);

template <>
__host__ __device__ inline float float_to_scalar<float>(float value) {
    return value;
}

template <>
__host__ __device__ inline __half float_to_scalar<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__host__ __device__ inline __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void reduction_partials_kernel(
    const T* __restrict__ input,
    float* __restrict__ partials,
    std::int64_t count,
    int partitions) {

    const int partition = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const std::int64_t begin = (count * partition) / partitions;
    const std::int64_t end = (count * (partition + 1)) / partitions;

    float local = 0.0f;
    for (std::int64_t index = begin + tid; index < end; index += blockDim.x) {
        local += scalar_to_float(input[index]);
    }

    extern __shared__ float shared[];
    shared[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }

    if (tid == 0) partials[partition] = shared[0];
}

__global__ void reduction_finalize_kernel(
    const float* __restrict__ partials,
    int partitions,
    float* __restrict__ output) {

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < partitions; ++i) total += partials[i];
        output[0] = total;
    }
}

template <typename T>
__global__ void splitk_lm_head_partials_kernel(
    const T* __restrict__ hidden,
    const T* __restrict__ weights,
    float* __restrict__ partials,
    int vocab,
    int hidden_size,
    int split_k) {

    const int token = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (token >= vocab) return;

    const int begin = (hidden_size * split) / split_k;
    const int end = (hidden_size * (split + 1)) / split_k;
    const T* weight_row = weights + static_cast<std::size_t>(token) * hidden_size;

    float local = 0.0f;
    for (int k = begin + tid; k < end; k += blockDim.x) {
        local = fmaf(
            scalar_to_float(hidden[k]),
            scalar_to_float(weight_row[k]),
            local);
    }

    extern __shared__ float shared[];
    shared[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }

    if (tid == 0) {
        partials[static_cast<std::size_t>(token) * split_k + split] = shared[0];
    }
}

__global__ void splitk_lm_head_finalize_kernel(
    const float* __restrict__ partials,
    float* __restrict__ logits,
    int vocab,
    int split_k) {

    const int token = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= vocab) return;

    float total = 0.0f;
    const float* row = partials + static_cast<std::size_t>(token) * split_k;
    for (int split = 0; split < split_k; ++split) total += row[split];
    logits[token] = total;
}

static std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float width");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static std::uint64_t ordered_float_bits(float value) {
    const std::uint32_t bits = float_bits(value);
    if (bits & 0x80000000u) {
        return static_cast<std::uint64_t>(~bits + 1u);
    }
    return static_cast<std::uint64_t>(bits | 0x80000000u);
}

static std::uint64_t ulp_distance(float lhs, float rhs) {
    if (std::isnan(lhs) || std::isnan(rhs)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const std::uint64_t a = ordered_float_bits(lhs);
    const std::uint64_t b = ordered_float_bits(rhs);
    return a >= b ? a - b : b - a;
}

static std::uint64_t fnv1a_hash(const std::vector<float>& values) {
    std::uint64_t hash = 1469598103934665603ull;
    for (float value : values) {
        const std::uint32_t bits = float_bits(value);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint8_t>((bits >> (8 * byte)) & 0xffu);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

struct DifferenceMetrics {
    bool bitwise_equal = true;
    double mismatch_fraction = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::uint64_t max_ulp = 0;
    int argmax_reference = -1;
    int argmax_test = -1;
    bool argmax_flip = false;
    std::size_t reference_nan_count = 0;
    std::size_t test_nan_count = 0;
    std::size_t reference_inf_count = 0;
    std::size_t test_inf_count = 0;
    std::uint64_t reference_hash = 0;
    std::uint64_t test_hash = 0;
};

static DifferenceMetrics compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& test,
    bool compute_argmax) {

    if (reference.size() != test.size() || reference.empty()) {
        throw std::invalid_argument("compare_outputs received incompatible vectors");
    }

    DifferenceMetrics metrics;
    std::size_t mismatch_count = 0;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float ref = reference[i];
        const float out = test[i];
        metrics.reference_nan_count += std::isnan(ref);
        metrics.test_nan_count += std::isnan(out);
        metrics.reference_inf_count += std::isinf(ref);
        metrics.test_inf_count += std::isinf(out);
        if (float_bits(ref) != float_bits(out)) ++mismatch_count;

        if (!std::isfinite(ref) || !std::isfinite(out)) continue;

        const double abs_error = std::abs(static_cast<double>(out) - static_cast<double>(ref));
        const double denominator = std::max(std::abs(static_cast<double>(ref)), 1.0e-20);
        const double rel_error = abs_error / denominator;

        metrics.max_abs = std::max(metrics.max_abs, abs_error);
        metrics.max_rel = std::max(metrics.max_rel, rel_error);
        metrics.max_ulp = std::max(metrics.max_ulp, ulp_distance(ref, out));
    }

    metrics.bitwise_equal = mismatch_count == 0;
    metrics.mismatch_fraction = static_cast<double>(mismatch_count) / reference.size();
    metrics.reference_hash = fnv1a_hash(reference);
    metrics.test_hash = fnv1a_hash(test);

    if (compute_argmax &&
        metrics.reference_nan_count == 0 && metrics.test_nan_count == 0 &&
        metrics.reference_inf_count == 0 && metrics.test_inf_count == 0) {
        metrics.argmax_reference = static_cast<int>(
            std::distance(reference.begin(), std::max_element(reference.begin(), reference.end())));
        metrics.argmax_test = static_cast<int>(
            std::distance(test.begin(), std::max_element(test.begin(), test.end())));
        metrics.argmax_flip = metrics.argmax_reference != metrics.argmax_test;
    }

    return metrics;
}

static std::string hex_hash(std::uint64_t value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return oss.str();
}

class CsvWriter {
public:
    CsvWriter(const std::string& path, const Options& opt)
        : stream_(path), exp_min_(opt.exp_min), exp_max_(opt.exp_max) {
        if (!stream_) throw std::runtime_error("cannot open CSV output: " + path);
        stream_
            << "experiment,dtype,trial,baseline_shape,test_shape,threads,allocation_tag,elements,"
            << "bitwise_equal,mismatch_fraction,max_abs,max_rel,max_ulp,"
            << "argmax_reference,argmax_test,argmax_flip,reference_hash,test_hash,"
            << "reference_nan_count,test_nan_count,reference_inf_count,test_inf_count,"
            << "exp_min,exp_max,accumulation\n";
        stream_ << std::setprecision(17);
    }

    void write(
        const std::string& experiment,
        const std::string& dtype,
        int trial,
        int baseline_shape,
        int test_shape,
        int threads,
        const std::string& allocation_tag,
        std::size_t elements,
        const DifferenceMetrics& m) {

        stream_
            << experiment << ','
            << dtype << ','
            << trial << ','
            << baseline_shape << ','
            << test_shape << ','
            << threads << ','
            << allocation_tag << ','
            << elements << ','
            << (m.bitwise_equal ? 1 : 0) << ','
            << m.mismatch_fraction << ','
            << m.max_abs << ','
            << m.max_rel << ','
            << m.max_ulp << ','
            << m.argmax_reference << ','
            << m.argmax_test << ','
            << (m.argmax_flip ? 1 : 0) << ','
            << hex_hash(m.reference_hash) << ','
            << hex_hash(m.test_hash) << ','
            << m.reference_nan_count << ',' << m.test_nan_count << ','
            << m.reference_inf_count << ',' << m.test_inf_count << ','
            << exp_min_ << ',' << exp_max_ << ",fp32\n";
    }

private:
    std::ofstream stream_;
    int exp_min_;
    int exp_max_;
};

static float generate_mixed_magnitude_value(
    std::mt19937_64& rng, int exponent_min, int exponent_max) {
    std::uniform_real_distribution<float> mantissa(0.5f, 1.0f);
    std::uniform_int_distribution<int> exponent(exponent_min, exponent_max);
    std::bernoulli_distribution sign(0.5);
    float value = std::ldexp(mantissa(rng), exponent(rng));
    return sign(rng) ? value : -value;
}


template <typename T>
static void convert_vector(const std::vector<float>& source, std::vector<T>& destination) {
    destination.resize(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        destination[i] = float_to_scalar<T>(source[i]);
    }
}

template <typename T>
static float run_reduction_once(
    const T* device_input,
    float* device_partials,
    float* device_output,
    std::int64_t count,
    int partitions,
    int threads) {

    reduction_partials_kernel<T><<<partitions, threads, threads * sizeof(float)>>>(
        device_input, device_partials, count, partitions);
    CUDA_CHECK(cudaGetLastError());

    reduction_finalize_kernel<<<1, 1>>>(device_partials, partitions, device_output);
    CUDA_CHECK(cudaGetLastError());

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

template <typename T>
static std::vector<float> run_splitk_gemm_once(
    const T* device_hidden,
    const T* device_weights,
    float* device_partials,
    float* device_logits,
    int vocab,
    int hidden_size,
    int split_k,
    int threads) {

    const dim3 grid(static_cast<unsigned int>(vocab), static_cast<unsigned int>(split_k), 1);
    splitk_lm_head_partials_kernel<T><<<grid, threads, threads * sizeof(float)>>>(
        device_hidden, device_weights, device_partials, vocab, hidden_size, split_k);
    CUDA_CHECK(cudaGetLastError());

    const int final_threads = 256;
    const int final_blocks = (vocab + final_threads - 1) / final_threads;
    splitk_lm_head_finalize_kernel<<<final_blocks, final_threads>>>(
        device_partials, device_logits, vocab, split_k);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> logits(vocab);
    CUDA_CHECK(cudaMemcpy(
        logits.data(), device_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return logits;
}

template <typename T>
static void run_reduction_experiment(
    const Options& opt,
    const std::string& dtype_name,
    CsvWriter& csv) {

    int baseline = opt.baseline_shape > 0 ? opt.baseline_shape : 64;
    std::vector<int> variants = opt.variants.empty()
        ? std::vector<int>{32, 64, 128, 256}
        : opt.variants;

    for (int shape : variants) {
        if (shape <= 0 || shape > 65535) {
            throw std::invalid_argument("reduction partition counts must be in [1, 65535]");
        }
    }
    if (baseline <= 0 || baseline > 65535) {
        throw std::invalid_argument("invalid reduction baseline partition count");
    }

    const int max_partitions = std::max(baseline, *std::max_element(variants.begin(), variants.end()));

    T* device_input = nullptr;
    float* device_partials = nullptr;
    float* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(&device_input, static_cast<std::size_t>(opt.reduction_size) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&device_partials, static_cast<std::size_t>(max_partitions) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));

    std::vector<float> host_float(static_cast<std::size_t>(opt.reduction_size));
    std::vector<T> host_typed;

    for (int trial = 0; trial < opt.trials; ++trial) {
        std::mt19937_64 rng(opt.seed + static_cast<std::uint64_t>(trial) * 0x9e3779b97f4a7c15ull);
        for (float& value : host_float) value = generate_mixed_magnitude_value(rng, opt.exp_min, opt.exp_max);
        convert_vector(host_float, host_typed);

        CUDA_CHECK(cudaMemcpy(
            device_input,
            host_typed.data(),
            host_typed.size() * sizeof(T),
            cudaMemcpyHostToDevice));

        const float reference_scalar = run_reduction_once(
            device_input,
            device_partials,
            device_output,
            opt.reduction_size,
            baseline,
            opt.threads);
        const std::vector<float> reference{reference_scalar};

        for (int shape : variants) {
            const float test_scalar = run_reduction_once(
                device_input,
                device_partials,
                device_output,
                opt.reduction_size,
                shape,
                opt.threads);
            const std::vector<float> test{test_scalar};
            const DifferenceMetrics metrics = compare_outputs(reference, test, false);

            csv.write(
                "reduction",
                dtype_name,
                trial,
                baseline,
                shape,
                opt.threads,
                opt.allocation_tag,
                1,
                metrics);
        }

        if ((trial + 1) % 10 == 0 || trial + 1 == opt.trials) {
            std::cerr << "[reduction/" << dtype_name << "] "
                      << (trial + 1) << '/' << opt.trials << " trials\n";
        }
    }

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_partials));
    CUDA_CHECK(cudaFree(device_input));
}

template <typename T>
static void run_gemm_experiment(
    const Options& opt,
    const std::string& dtype_name,
    CsvWriter& csv) {

    int baseline = opt.baseline_shape > 0 ? opt.baseline_shape : 1;
    std::vector<int> variants = opt.variants.empty()
        ? std::vector<int>{1, 2, 4, 8, 16}
        : opt.variants;

    for (int split_k : variants) {
        if (split_k <= 0 || split_k > opt.hidden || split_k > 65535) {
            throw std::invalid_argument("split-K factors must be positive and <= hidden size");
        }
    }
    if (baseline <= 0 || baseline > opt.hidden || baseline > 65535) {
        throw std::invalid_argument("invalid GEMM baseline split-K factor");
    }

    const int max_split = std::max(baseline, *std::max_element(variants.begin(), variants.end()));
    const std::size_t weight_count = static_cast<std::size_t>(opt.vocab) * opt.hidden;
    const std::size_t partial_count = static_cast<std::size_t>(opt.vocab) * max_split;

    T* device_hidden = nullptr;
    T* device_weights = nullptr;
    float* device_partials = nullptr;
    float* device_logits = nullptr;

    CUDA_CHECK(cudaMalloc(&device_hidden, static_cast<std::size_t>(opt.hidden) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&device_weights, weight_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&device_partials, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_logits, static_cast<std::size_t>(opt.vocab) * sizeof(float)));

    // Keep the LM-head weight matrix fixed across trials and vary the hidden state.
    std::mt19937_64 weight_rng(opt.seed ^ 0xd1b54a32d192ed03ull);
    std::vector<float> host_weights_float(weight_count);
    for (float& value : host_weights_float) {
        value = generate_mixed_magnitude_value(weight_rng, opt.exp_min, opt.exp_max);
    }
    std::vector<T> host_weights;
    convert_vector(host_weights_float, host_weights);
    CUDA_CHECK(cudaMemcpy(
        device_weights,
        host_weights.data(),
        host_weights.size() * sizeof(T),
        cudaMemcpyHostToDevice));

    std::vector<float> host_hidden_float(static_cast<std::size_t>(opt.hidden));
    std::vector<T> host_hidden;

    for (int trial = 0; trial < opt.trials; ++trial) {
        std::mt19937_64 hidden_rng(
            opt.seed + static_cast<std::uint64_t>(trial) * 0x94d049bb133111ebull);
        for (float& value : host_hidden_float) {
            value = generate_mixed_magnitude_value(hidden_rng, opt.exp_min, opt.exp_max);
        }
        convert_vector(host_hidden_float, host_hidden);
        CUDA_CHECK(cudaMemcpy(
            device_hidden,
            host_hidden.data(),
            host_hidden.size() * sizeof(T),
            cudaMemcpyHostToDevice));

        const std::vector<float> reference = run_splitk_gemm_once(
            device_hidden,
            device_weights,
            device_partials,
            device_logits,
            opt.vocab,
            opt.hidden,
            baseline,
            opt.threads);

        for (int split_k : variants) {
            const std::vector<float> test = run_splitk_gemm_once(
                device_hidden,
                device_weights,
                device_partials,
                device_logits,
                opt.vocab,
                opt.hidden,
                split_k,
                opt.threads);
            const DifferenceMetrics metrics = compare_outputs(reference, test, true);

            csv.write(
                "splitk_lm_head_gemm",
                dtype_name,
                trial,
                baseline,
                split_k,
                opt.threads,
                opt.allocation_tag,
                static_cast<std::size_t>(opt.vocab),
                metrics);
        }

        if ((trial + 1) % 10 == 0 || trial + 1 == opt.trials) {
            std::cerr << "[gemm/" << dtype_name << "] "
                      << (trial + 1) << '/' << opt.trials << " trials\n";
        }
    }

    CUDA_CHECK(cudaFree(device_logits));
    CUDA_CHECK(cudaFree(device_partials));
    CUDA_CHECK(cudaFree(device_weights));
    CUDA_CHECK(cudaFree(device_hidden));
}

template <typename T>
static void run_selected_experiments(
    const Options& opt,
    const std::string& dtype_name,
    CsvWriter& csv) {

    if (opt.experiment == "reduction" || opt.experiment == "both") {
        run_reduction_experiment<T>(opt, dtype_name, csv);
    }
    if (opt.experiment == "gemm" || opt.experiment == "both") {
        run_gemm_experiment<T>(opt, dtype_name, csv);
    }
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        CsvWriter csv(opt.csv_path, opt);

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp property{};
        CUDA_CHECK(cudaGetDeviceProperties(&property, device));

        std::cerr << "GPU: " << property.name << "\n"
                  << "Experiment: " << opt.experiment << "\n"
                  << "Trials: " << opt.trials << "\n"
                  << "Threads: " << opt.threads << "\n"
                  << "Input exponent range: [" << opt.exp_min << ", " << opt.exp_max << "]\n"
                  << "Accumulation: FP32\n"
                  << "CSV: " << opt.csv_path << "\n";

        if (opt.dtype == "fp32" || opt.dtype == "all") {
            run_selected_experiments<float>(opt, "fp32", csv);
        }
        if (opt.dtype == "fp16" || opt.dtype == "all") {
            run_selected_experiments<__half>(opt, "fp16", csv);
        }
        if (opt.dtype == "bf16" || opt.dtype == "all") {
            run_selected_experiments<__nv_bfloat16>(opt, "bf16", csv);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        std::cerr << "Done. Results written to " << opt.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
