/*
 * RoPE + GQA Flash Attention — Correctness + Performance Test
 *
 * Calls miopenRopeGqaForward() which fuses:
 *   1) RoPE (rotary position embedding) on Q and K   — JIT kernel
 *   2) GQA flash attention (CK Tile FMHA)            — compiled kernel
 *
 * Usage:
 *   test_rope_gqa.exe batch seqlen_q seqlen_k nhead_q nhead_k hdim [data_dir]
 *
 * If data_dir is provided, loads golden reference and validates.
 * Otherwise runs benchmark with random data.
 */

#define MIOPEN_BETA_API 1
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <random>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        std::cerr << "HIP error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define MIOPEN_CHECK(call) do { \
    miopenStatus_t st = (call); \
    if (st != miopenStatusSuccess) { \
        std::cerr << "MIOpen error " << st << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

template <typename T>
bool read_binary(const std::string& path, std::vector<T>& buf, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    if (static_cast<size_t>(f.tellg()) != count * sizeof(T)) return false;
    f.seekg(0);
    buf.resize(count);
    f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(T));
    return true;
}

std::map<std::string, std::string> read_meta(const std::string& path)
{
    std::map<std::string, std::string> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos)
            m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

void generate_cos_sin(std::vector<__half>& cos_buf, std::vector<__half>& sin_buf,
                      int max_seqlen, int rotary_dim)
{
    int half_rot = rotary_dim / 2;
    cos_buf.resize(max_seqlen * half_rot);
    sin_buf.resize(max_seqlen * half_rot);
    for (int s = 0; s < max_seqlen; s++) {
        for (int d = 0; d < half_rot; d++) {
            double theta = (double)s / std::pow(10000.0, 2.0 * d / rotary_dim);
            cos_buf[s * half_rot + d] = __float2half((float)std::cos(theta));
            sin_buf[s * half_rot + d] = __float2half((float)std::sin(theta));
        }
    }
}

int main(int argc, char* argv[])
{
    int batch    = (argc > 1) ? std::atoi(argv[1]) : 2;
    int seqlen_q = (argc > 2) ? std::atoi(argv[2]) : 256;
    int seqlen_k = (argc > 3) ? std::atoi(argv[3]) : 256;
    int nhead_q  = (argc > 4) ? std::atoi(argv[4]) : 32;
    int nhead_k  = (argc > 5) ? std::atoi(argv[5]) : 8;
    int hdim     = (argc > 6) ? std::atoi(argv[6]) : 128;
    std::string data_dir = (argc > 7) ? argv[7] : "";
    int rotary_dim = hdim;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "MIOpen RoPE + GQA Flash Attention Test" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "GPU: " << prop.name << " (arch: " << prop.gcnArchName << ")" << std::endl;

    if (!data_dir.empty()) {
        auto meta = read_meta(data_dir + "/gqa_meta.txt");
        if (!meta.empty()) {
            batch      = std::stoi(meta["batch"]);
            seqlen_q   = std::stoi(meta["seqlen_q"]);
            seqlen_k   = std::stoi(meta["seqlen_k"]);
            nhead_q    = std::stoi(meta["nhead_q"]);
            nhead_k    = std::stoi(meta["nhead_k"]);
            hdim       = std::stoi(meta["hdim"]);
            rotary_dim = meta.count("rotary_dim") ? std::stoi(meta["rotary_dim"]) : hdim;
        }
    }

    int ratio = nhead_q / nhead_k;
    int max_seqlen = std::max(seqlen_q, seqlen_k);
    int half_rot = rotary_dim / 2;

    std::cout << "  batch=" << batch << " seqlen_q=" << seqlen_q << " seqlen_k=" << seqlen_k
              << " nhead_q=" << nhead_q << " nhead_k=" << nhead_k
              << " hdim=" << hdim << " rotary_dim=" << rotary_dim
              << " ratio=" << ratio << std::endl << std::endl;

    size_t q_cnt   = (size_t)batch * nhead_q * seqlen_q * hdim;
    size_t k_cnt   = (size_t)batch * nhead_k * seqlen_k * hdim;
    size_t v_cnt   = k_cnt;
    size_t o_cnt   = q_cnt;
    size_t cos_cnt = (size_t)max_seqlen * half_rot;

    // MIOpen handle
    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    // Device memory
    void *d_q, *d_k, *d_v, *d_o, *d_cos, *d_sin, *d_q_orig, *d_k_orig;
    HIP_CHECK(hipMalloc(&d_q,      q_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_k,      k_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_v,      v_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_o,      o_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_cos,    cos_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_sin,    cos_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_q_orig, q_cnt * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_k_orig, k_cnt * sizeof(__half)));

    bool has_ref = false;
    std::vector<__half> h_o_ref;

    if (!data_dir.empty()) {
        std::vector<__half> h_q, h_k, h_v, h_cos, h_sin;

        if (!read_binary(data_dir + "/gqa_Q.bin", h_q, q_cnt) ||
            !read_binary(data_dir + "/gqa_K.bin", h_k, k_cnt) ||
            !read_binary(data_dir + "/gqa_V.bin", h_v, v_cnt) ||
            !read_binary(data_dir + "/gqa_cos.bin", h_cos, cos_cnt) ||
            !read_binary(data_dir + "/gqa_sin.bin", h_sin, cos_cnt)) {
            std::cerr << "Failed to read data files from " << data_dir << std::endl;
            return 1;
        }

        has_ref = read_binary(data_dir + "/gqa_O_ref.bin", h_o_ref, o_cnt);

        HIP_CHECK(hipMemcpy(d_q_orig, h_q.data(), q_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_k_orig, h_k.data(), k_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_v,  h_v.data(),  v_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_cos, h_cos.data(), cos_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sin, h_sin.data(), cos_cnt * sizeof(__half), hipMemcpyHostToDevice));
        std::cout << "  Loaded data from " << data_dir << "/" << std::endl;
    } else {
        std::vector<__half> h_q(q_cnt), h_k(k_cnt), h_v(v_cnt);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& x : h_q) x = __float2half(dist(rng));
        for (auto& x : h_k) x = __float2half(dist(rng));
        for (auto& x : h_v) x = __float2half(dist(rng));

        std::vector<__half> h_cos, h_sin;
        generate_cos_sin(h_cos, h_sin, max_seqlen, rotary_dim);

        HIP_CHECK(hipMemcpy(d_q_orig, h_q.data(), q_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_k_orig, h_k.data(), k_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_v,  h_v.data(),  v_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_cos, h_cos.data(), cos_cnt * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sin, h_sin.data(), cos_cnt * sizeof(__half), hipMemcpyHostToDevice));
        std::cout << "  Using random data (no validation)" << std::endl;
    }

    // RoPE modifies Q/K in-place, so we need to restore before each call
    auto run_once = [&]() {
        HIP_CHECK(hipMemcpy(d_q, d_q_orig, q_cnt * sizeof(__half), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_k, d_k_orig, k_cnt * sizeof(__half), hipMemcpyDeviceToDevice));
        MIOPEN_CHECK(miopenRopeGqaForward(
            handle, batch, seqlen_q, seqlen_k, nhead_q, nhead_k,
            hdim, rotary_dim, d_q, d_k, d_v, d_o, d_cos, d_sin));
    };

    // Warmup
    std::cout << "  Warmup..." << std::flush;
    for (int i = 0; i < 3; i++) run_once();
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << " done" << std::endl;

    // Benchmark
    int repeat = 20;
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));

    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < repeat; i++) run_once();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));

    float ms_total = 0;
    HIP_CHECK(hipEventElapsedTime(&ms_total, t0, t1));
    float ms_avg = ms_total / repeat;

    double flops = 4.0 * batch * nhead_q * seqlen_q * (double)seqlen_k * hdim;
    double gflops = flops / (ms_avg * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n  === Performance ===" << std::endl;
    std::cout << "  Median: " << ms_avg << " ms, " << gflops << " GFLOPS" << std::endl;

    // Validation
    if (has_ref) {
        run_once();
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<__half> h_o(o_cnt);
        HIP_CHECK(hipMemcpy(h_o.data(), d_o, o_cnt * sizeof(__half), hipMemcpyDeviceToHost));

        int errors = 0;
        double max_diff = 0;
        for (size_t i = 0; i < o_cnt; i++) {
            float ref = __half2float(h_o_ref[i]);
            float act = __half2float(h_o[i]);
            float diff = std::fabs(ref - act);
            float tol = std::fabs(ref) * 0.02f + 0.01f;
            max_diff = std::max(max_diff, (double)diff);
            if (diff > tol) errors++;
        }

        std::cout << "\n  === Validation ===" << std::endl;
        std::cout << "  Elements: " << o_cnt << ", Errors: " << errors << std::endl;
        std::cout << "  Max abs diff: " << max_diff << std::endl;

        for (int i = 0; i < std::min((int)o_cnt, 5); i++) {
            float ref = __half2float(h_o_ref[i]);
            float act = __half2float(h_o[i]);
            std::cout << "    [" << i << "] GPU=" << std::setprecision(6) << act
                      << "  Ref=" << ref << "  diff=" << std::fabs(ref - act) << std::endl;
        }

        bool pass = (errors == 0);
        std::cout << "  Result: " << (pass ? "PASSED" : "FAILED") << std::endl;

        std::cout << "\n======================================" << std::endl;
        std::cout << "Overall: " << (pass ? "ALL PASSED" : "SOME FAILED") << std::endl;

        HIP_CHECK(hipEventDestroy(t0));
        HIP_CHECK(hipEventDestroy(t1));
        hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o);
        hipFree(d_cos); hipFree(d_sin); hipFree(d_q_orig); hipFree(d_k_orig);
        miopenDestroy(handle);
        return pass ? 0 : 1;
    }

    std::cout << "\n======================================" << std::endl;
    std::cout << "Benchmark complete (no validation data)" << std::endl;

    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o);
    hipFree(d_cos); hipFree(d_sin); hipFree(d_q_orig); hipFree(d_k_orig);
    miopenDestroy(handle);
    return 0;
}
