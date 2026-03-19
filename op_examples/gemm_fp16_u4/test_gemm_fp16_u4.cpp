// ============================================================
// MIOpen GemmFp16U4 (Fused GEMM + Dequantization) Verification Test
//
// Tests the miopenGemmFp16U4Forward() API which performs:
//   C[M×N] = A[M×K] × dequant(B_packed[N×K/2])^T
//
// A:        FP16 col-major (M × K), stride lda = M
// B_packed: uint4 packed (N × K/2), each byte = 2 values (low nibble first)
// scales:   FP16 (N × num_groups_k), per-column per-group
// zeros:    FP16 (N × num_groups_k), per-column per-group zero point
// C:        FP16 col-major (M × N), stride ldc = M
//
// Workflow:
//   1) python3 gen_gemm_fp16_u4_data.py MxKxN --group-size GS --dir data
//   2) make && make run
//
// Build: make
// Run:   make run
// ============================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define MIOPEN_BETA_API 1
#include <miopen/miopen.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <functional>

#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = (call);                                            \
        if(err != hipSuccess)                                               \
        {                                                                   \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \""                           \
                      << hipGetErrorString(err) << "\"" << std::endl;       \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

#define MIOPEN_CHECK(call)                                                  \
    do                                                                      \
    {                                                                       \
        miopenStatus_t s = (call);                                          \
        if(s != miopenStatusSuccess)                                        \
        {                                                                   \
            std::cerr << "MIOpen error at " << __FILE__ << ":" << __LINE__  \
                      << " status=" << s << std::endl;                      \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

template <typename T>
static bool readBin(const std::string& path, std::vector<T>& data, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open())
        return false;
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return f.good();
}

// ============================================================
// Benchmark helpers
// ============================================================
static int calibrateIters(double warmup_ms, int warmup_count,
                          double target_ms = 2000.0, int lo = 5, int hi = 200)
{
    double per_iter = warmup_ms / warmup_count;
    if(per_iter <= 0) return lo;
    return std::max(lo, std::min(hi, static_cast<int>(target_ms / per_iter)));
}

constexpr int NROUNDS = 5;

struct MeasureResult {
    double median_ms;
    double min_ms;
    double max_ms;
};

static MeasureResult measureMedian(hipStream_t stream, int niters,
                                   const std::function<void()>& launch)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    std::vector<float> round_ms(NROUNDS);
    for(int r = 0; r < NROUNDS; r++)
    {
        hipEventRecord(ev0, stream);
        for(int i = 0; i < niters; i++)
            launch();
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&round_ms[r], ev0, ev1);
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    std::sort(round_ms.begin(), round_ms.end());
    return {round_ms[NROUNDS / 2], round_ms[0], round_ms[NROUNDS - 1]};
}

bool test_gemm_fp16_u4(int M, int N, int K, int group_size, const std::string& data_dir)
{
    int num_groups_k = K / group_size;

    std::cout << "\n=== Test GemmFp16U4 M=" << M << " N=" << N << " K=" << K
              << " group_size=" << group_size << " ===" << std::endl;

    std::string fA = data_dir + "/gemm_fp16_u4_A.bin";
    std::string fB = data_dir + "/gemm_fp16_u4_B_packed.bin";
    std::string fS = data_dir + "/gemm_fp16_u4_scales.bin";
    std::string fZ = data_dir + "/gemm_fp16_u4_zeros.bin";
    std::string fC = data_dir + "/gemm_fp16_u4_C_ref.bin";

    std::vector<__half>  h_A;
    std::vector<uint8_t> h_B_packed;
    std::vector<__half>  h_scales;
    std::vector<__half>  h_zeros;

    size_t countA = static_cast<size_t>(M) * K;
    size_t countB = static_cast<size_t>(N) * K / 2;
    size_t countS = static_cast<size_t>(N) * num_groups_k;
    size_t countZ = static_cast<size_t>(N) * num_groups_k;
    size_t countC = static_cast<size_t>(M) * N;

    if(!readBin(fA, h_A, countA) ||
       !readBin(fB, h_B_packed, countB) ||
       !readBin(fS, h_scales, countS) ||
       !readBin(fZ, h_zeros, countZ))
    {
        std::cerr << "  ERROR: Failed to read input data files from " << data_dir << "/" << std::endl;
        std::cerr << "  Run: python3 gen_gemm_fp16_u4_data.py "
                  << M << "x" << K << "x" << N
                  << " --group-size " << group_size
                  << " --dir " << data_dir << std::endl;
        return false;
    }
    std::cout << "  Loaded input data from " << data_dir << "/" << std::endl;

    std::vector<__half> h_C_ref;
    bool has_ref = readBin(fC, h_C_ref, countC);
    if(!has_ref)
        std::cout << "  WARNING: No Python reference file (" << fC << "), skipping verification." << std::endl;

    __half*  d_A        = nullptr;
    uint8_t* d_B_packed = nullptr;
    __half*  d_scales   = nullptr;
    __half*  d_zeros    = nullptr;
    __half*  d_C        = nullptr;

    size_t size_A      = countA * sizeof(__half);
    size_t size_B      = countB;
    size_t size_scales = countS * sizeof(__half);
    size_t size_zeros  = countZ * sizeof(__half);
    size_t size_C      = countC * sizeof(__half);

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B_packed, size_B));
    HIP_CHECK(hipMalloc(&d_scales, size_scales));
    HIP_CHECK(hipMalloc(&d_zeros, size_zeros));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_packed, h_B_packed.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, h_scales.data(), size_scales, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_zeros, h_zeros.data(), size_zeros, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, size_C));

    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    hipStream_t stream;
    MIOPEN_CHECK(miopenGetStream(handle, &stream));

    std::cout << "  Warmup..." << std::flush;
    auto tw0 = std::chrono::steady_clock::now();
    miopenStatus_t status = miopenStatusSuccess;
    for(int w = 0; w < 3; w++)
    {
        status = miopenGemmFp16U4Forward(
            handle, M, N, K,
            d_A, M,
            d_B_packed,
            d_scales, d_zeros,
            group_size, num_groups_k,
            d_C, M);
        if(status != miopenStatusSuccess) break;
    }
    hipStreamSynchronize(stream);
    auto tw1 = std::chrono::steady_clock::now();
    double warmup_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(tw1 - tw0).count() / 1000.0;

    if(status != miopenStatusSuccess)
    {
        std::cout << " FAILED (status=" << status << ")" << std::endl;
        hipFree(d_A);
        hipFree(d_B_packed);
        hipFree(d_scales);
        hipFree(d_zeros);
        hipFree(d_C);
        miopenDestroy(handle);
        return false;
    }

    int niters = calibrateIters(warmup_ms, 3);
    std::cout << " OK (" << std::fixed << std::setprecision(2) << warmup_ms
              << " ms), iters=" << niters << std::endl;

    std::cout << "  Benchmarking (" << NROUNDS << " rounds x " << niters << " iters)..."
              << std::flush;
    auto mr = measureMedian(stream, niters, [&]() {
        miopenGemmFp16U4Forward(
            handle, M, N, K,
            d_A, M,
            d_B_packed,
            d_scales, d_zeros,
            group_size, num_groups_k,
            d_C, M);
    });

    double avg_ms    = mr.median_ms / niters;
    double gflops    = (2.0 * M * N * K) / (avg_ms * 1e6);
    double mem_bytes = static_cast<double>(countA) * 2 + static_cast<double>(countB)
                     + static_cast<double>(countS) * 2 + static_cast<double>(countZ) * 2
                     + static_cast<double>(countC) * 2;
    double bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);
    double range_pct = (mr.max_ms - mr.min_ms) / mr.median_ms * 100.0;

    std::cout << " done" << std::endl;
    std::cout << "\n  === Performance ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Median: " << avg_ms << " ms, "
              << gflops << " GFLOPS, "
              << bw_gbs << " GB/s" << std::endl;
    std::cout << "  Range:  " << mr.min_ms / niters << " ~ " << mr.max_ms / niters
              << " ms  (jitter " << std::setprecision(1) << range_pct << "%)" << std::endl;

    std::vector<__half> h_C(countC);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    bool pass = true;

    if(has_ref)
    {
        int errors      = 0;
        int total        = M * N;
        float max_diff  = 0.0f;
        float max_rdiff = 0.0f;

        for(int i = 0; i < total; i++)
        {
            float gpu_val = __half2float(h_C[i]);
            float ref_val = __half2float(h_C_ref[i]);
            float diff    = std::fabs(gpu_val - ref_val);
            float rdiff   = (std::fabs(ref_val) > 1e-6f) ? diff / std::fabs(ref_val) : diff;

            if(diff > max_diff) max_diff = diff;
            if(rdiff > max_rdiff) max_rdiff = rdiff;

            float tol = std::fabs(ref_val) * 0.05f + 0.1f;
            if(diff > tol)
                errors++;
        }

        std::cout << "\n  === GPU vs Python Reference ===" << std::endl;
        std::cout << "  Verified " << total << " elements, " << errors << " errors" << std::endl;
        std::cout << "  Max abs diff: " << max_diff << ", max rel diff: "
                  << std::fixed << std::setprecision(4) << (max_rdiff * 100.0f) << "%" << std::endl;

        std::cout << "  Sample C values (GPU vs Python):" << std::endl;
        for(int i = 0; i < 5 && i < total; i++)
        {
            float gpu_val = __half2float(h_C[i]);
            float ref_val = __half2float(h_C_ref[i]);
            std::cout << "    [" << i << "] GPU=" << std::setprecision(6) << gpu_val
                      << "  Ref=" << ref_val
                      << "  diff=" << std::fabs(gpu_val - ref_val) << std::endl;
        }

        pass = (errors == 0);
        std::cout << "  Result: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }
    else
    {
        std::cout << "  GPU output sample:" << std::endl;
        int total = M * N;
        for(int i = 0; i < 5 && i < total; i++)
            std::cout << "    [" << i << "] = " << __half2float(h_C[i]) << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B_packed);
    hipFree(d_scales);
    hipFree(d_zeros);
    hipFree(d_C);
    miopenDestroy(handle);

    return pass;
}

int main(int argc, char* argv[])
{
    std::cout << "MIOpen GemmFp16U4 (Fused GEMM + Dequantization) Verification" << std::endl;
    std::cout << "=============================================================" << std::endl;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (arch: " << prop.gcnArchName << ")" << std::endl;

    std::string arch(prop.gcnArchName);
    if(arch.find("gfx11") == std::string::npos && arch.find("gfx12") == std::string::npos)
    {
        std::cerr << "WARNING: This kernel requires RDNA3+ (gfx11xx/gfx12xx). "
                  << "Current arch: " << arch << std::endl;
    }

    int M = 128, N = 128, K = 128, gs = 128;
    std::string data_dir = "data";

    if(argc >= 2)
    {
        if(sscanf(argv[1], "%dx%dx%d", &M, &K, &N) != 3)
        {
            std::cerr << "Usage: " << argv[0] << " [MxKxN] [group_size] [data_dir]" << std::endl;
            return 1;
        }
    }
    if(argc >= 3) gs = atoi(argv[2]);
    if(argc >= 4) data_dir = argv[3];

    std::cout << "Data dir: " << data_dir << std::endl;

    bool all_pass = true;
    all_pass &= test_gemm_fp16_u4(M, N, K, gs, data_dir);

    std::cout << "\n=============================================================" << std::endl;
    std::cout << "Overall: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << std::endl;

    return all_pass ? 0 : 1;
}
