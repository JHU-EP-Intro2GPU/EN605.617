// Martingale Posterior via Predictive Resampling
//
// Implements Fong, Holmes & Walker (2021) on GPU.
// Computes posterior distributions for the mean and
// variance of a dataset without specifying a
// likelihood or prior.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>

// ============================================================
// Configuration
// ============================================================

#define N_RUNS      100000
#define N_OBS       50
#define N_PREDICT   500
#define ALPHA       1.0f
#define MU0         0.0f
#define SIGMA0      10.0f
#define TRUE_MU     3.0f
#define TRUE_SIGMA  2.0f
#define N_BINS      100
#define SEED        42
#define BLOCK_SIZE  256


// ============================================================
// GPU Kernels (cuRAND)
// ============================================================

// Initialize one cuRAND state per resampling run.
__global__ void init_states(
    curandState_t* states, unsigned int seed,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        curand_init(seed, tid, 0, &states[tid]);
}

// Polya urn draw: with prob n/(n+alpha), resample
// from data; otherwise draw from base measure.
__device__ float polya_draw(
    const float* obs, int n_obs,
    const float* pred, int n_pred,
    float alpha, curandState_t* st)
{
    int n = n_obs + n_pred;
    float u = curand_uniform(st);

    if (u < (float)n / (n + alpha)) {
        int idx = curand(st) % n;
        return (idx < n_obs) ? obs[idx]
                             : pred[idx - n_obs];
    }
    return MU0 + SIGMA0 * curand_normal(st);
}

// Predictive resampling — mean statistic.
// Each thread independently resamples N_PREDICT
// future observations and computes the overall mean.
__global__ void resample_mean(
    const float* obs, float* pred,
    float* stats, curandState_t* states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_RUNS) return;

    curandState_t st = states[tid];
    float* my = pred + (long long)tid * N_PREDICT;

    float sum = 0.0f;
    for (int i = 0; i < N_OBS; i++)
        sum += obs[i];

    for (int t = 0; t < N_PREDICT; t++) {
        float y = polya_draw(
            obs, N_OBS, my, t, ALPHA, &st);
        my[t] = y;
        sum += y;
    }

    stats[tid] = sum / (N_OBS + N_PREDICT);
    states[tid] = st;
}

// Predictive resampling — variance statistic.
// Uses Welford's online algorithm.
__global__ void resample_variance(
    const float* obs, float* pred,
    float* stats, curandState_t* states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_RUNS) return;

    curandState_t st = states[tid];
    float* my = pred + (long long)tid * N_PREDICT;

    float mean = 0.0f, m2 = 0.0f;
    for (int i = 0; i < N_OBS; i++) {
        float d = obs[i] - mean;
        mean += d / (i + 1);
        m2 += d * (obs[i] - mean);
    }

    for (int t = 0; t < N_PREDICT; t++) {
        float y = polya_draw(
            obs, N_OBS, my, t, ALPHA, &st);
        my[t] = y;
        int k = N_OBS + t + 1;
        float d = y - mean;
        mean += d / k;
        m2 += d * (y - mean);
    }

    stats[tid] = m2 / (N_OBS + N_PREDICT);
    states[tid] = st;
}

// ============================================================
// Thrust Functors
// ============================================================

struct SquareDev {
    float mean;
    SquareDev(float m) : mean(m) {}
    __host__ __device__
    float operator()(float x) const {
        float d = x - mean;
        return d * d;
    }
};

struct NormDensity {
    float total, bw;
    NormDensity(float t, float b) : total(t), bw(b) {}
    __host__ __device__
    float operator()(int c) const {
        return (float)c / (total * bw);
    }
};

// ============================================================
// Thrust Statistics
// ============================================================

// Compute mean of device vector.
float thrust_mean(thrust::device_vector<float>& v) {
    float s = thrust::reduce(
        v.begin(), v.end(), 0.0f,
        thrust::plus<float>());
    return s / (float)v.size();
}

// Compute std dev of device vector.
float thrust_std(
    thrust::device_vector<float>& v, float mean)
{
    float var = thrust::transform_reduce(
        v.begin(), v.end(), SquareDev(mean),
        0.0f, thrust::plus<float>());
    return sqrtf(var / (float)(v.size() - 1));
}

// Extract quantile from sorted device vector.
float thrust_quantile(
    thrust::device_vector<float>& v, float q)
{
    int idx = (int)(q * (v.size() - 1));
    float val;
    thrust::copy_n(v.begin() + idx, 1, &val);
    return val;
}

// ============================================================
// Histogram (Thrust)
// ============================================================

// Build histogram and write to CSV.
// Write density values to CSV file.
void write_density_csv(
    thrust::host_vector<float>& h_dens,
    float lo, float bw, const char* path)
{
    FILE* fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fprintf(fp, "bin_center,density\n");
    for (int i = 0; i < N_BINS; i++) {
        float c = lo + (i + 0.5f) * bw;
        fprintf(fp, "%.6f,%.6f\n", c, h_dens[i]);
    }
    fclose(fp);
    printf("  %s\n", path);
}

// Build histogram via lower_bound and write CSV.
void write_histogram(
    thrust::device_vector<float>& sorted,
    float lo, float hi, const char* path)
{
    int n = (int)sorted.size();
    float bw = (hi - lo) / N_BINS;

    // Bin edges via lower_bound
    thrust::device_vector<float> edges(N_BINS + 1);
    thrust::sequence(edges.begin(), edges.end());
    thrust::transform(edges.begin(), edges.end(),
        edges.begin(),
        [lo, bw] __device__ (float i) {
            return lo + i * bw;
        });

    thrust::device_vector<int> cum(N_BINS + 1);
    thrust::lower_bound(sorted.begin(), sorted.end(),
        edges.begin(), edges.end(), cum.begin());

    // Per-bin counts and normalize to density
    thrust::host_vector<int> h_cum = cum;
    thrust::device_vector<int> counts(N_BINS);
    for (int i = 0; i < N_BINS; i++)
        counts[i] = h_cum[i + 1] - h_cum[i];

    thrust::device_vector<float> dens(N_BINS);
    thrust::transform(counts.begin(), counts.end(),
        dens.begin(), NormDensity((float)n, bw));

    thrust::host_vector<float> h_dens = dens;
    write_density_csv(h_dens, lo, bw, path);
}

// ============================================================
// Summary Output
// ============================================================

// Compute and print posterior summary for one stat.
void print_posterior(
    thrust::device_vector<float>& v,
    const char* name, float true_val,
    float lo, float hi, const char* csv)
{
    thrust::sort(v.begin(), v.end());

    float mean = thrust_mean(v);
    float std  = thrust_std(v, mean);
    float q025 = thrust_quantile(v, 0.025f);
    float med  = thrust_quantile(v, 0.5f);
    float q975 = thrust_quantile(v, 0.975f);

    printf("  %s posterior:\n", name);
    printf("    mean   = %.4f\n", mean);
    printf("    std    = %.4f\n", std);
    printf("    median = %.4f\n", med);
    printf("    95%% CI = [%.4f, %.4f]\n", q025, q975);
    printf("    true   = %.4f\n\n", true_val);

    write_histogram(v, lo, hi, csv);
}

// Write combined summary CSV.
void write_summary(
    thrust::device_vector<float>& dm,
    thrust::device_vector<float>& dv)
{
    FILE* fp = fopen("data/summary.csv", "w");
    if (!fp) return;
    fprintf(fp,
        "statistic,mean,std,q025,median,"
        "q975,true_value\n");

    float mm = thrust_mean(dm);
    float ms = thrust_std(dm, mm);
    fprintf(fp, "mean,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f\n", mm, ms,
        thrust_quantile(dm, 0.025f),
        thrust_quantile(dm, 0.5f),
        thrust_quantile(dm, 0.975f),
        TRUE_MU);

    float vm = thrust_mean(dv);
    float vs = thrust_std(dv, vm);
    fprintf(fp, "variance,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f\n", vm, vs,
        thrust_quantile(dv, 0.025f),
        thrust_quantile(dv, 0.5f),
        thrust_quantile(dv, 0.975f),
        TRUE_SIGMA * TRUE_SIGMA);

    fclose(fp);
    printf("  data/summary.csv\n");
}

// ============================================================
// Main
// ============================================================

// Generate observed data using cuRAND host API.
void generate_obs(
    thrust::device_vector<float>& d_obs)
{
    curandGenerator_t gen;
    curandCreateGenerator(
        &gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, SEED);
    curandGenerateNormal(gen,
        thrust::raw_pointer_cast(d_obs.data()),
        N_OBS, TRUE_MU, TRUE_SIGMA);
    curandDestroyGenerator(gen);
}

// Run predictive resampling for mean statistic.
void run_mean_posterior(
    thrust::device_vector<float>& d_obs,
    float* d_pred,
    thrust::device_vector<float>& d_m,
    curandState_t* d_states, int grid)
{
    init_states<<<grid, BLOCK_SIZE>>>(
        d_states, SEED, N_RUNS);
    cudaDeviceSynchronize();
    printf("Resampling mean posterior...\n");
    resample_mean<<<grid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_obs.data()),
        d_pred,
        thrust::raw_pointer_cast(d_m.data()),
        d_states);
    cudaDeviceSynchronize();
}

// Run predictive resampling for variance statistic.
void run_var_posterior(
    thrust::device_vector<float>& d_obs,
    float* d_pred,
    thrust::device_vector<float>& d_v,
    curandState_t* d_states, int grid)
{
    init_states<<<grid, BLOCK_SIZE>>>(
        d_states, SEED + 999, N_RUNS);
    cudaDeviceSynchronize();
    printf("Resampling variance posterior...\n\n");
    resample_variance<<<grid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_obs.data()),
        d_pred,
        thrust::raw_pointer_cast(d_v.data()),
        d_states);
    cudaDeviceSynchronize();
}

// Print header with experiment parameters.
void print_header() {
    printf("Martingale Posterior via "
           "Predictive Resampling\n");
    printf("=================================="
           "==========\n\n");
    printf("  N(%.1f, %.1f^2), n=%d, runs=%d, "
           "T=%d, alpha=%.1f\n\n",
           TRUE_MU, TRUE_SIGMA, N_OBS,
           N_RUNS, N_PREDICT, ALPHA);
}

// Print posterior summaries, histograms, CSV.
void print_results(
    thrust::device_vector<float>& d_m,
    thrust::device_vector<float>& d_v,
    float obs_mean, float obs_std)
{
    printf("Results:\n");
    print_posterior(d_m, "Mean", TRUE_MU,
        obs_mean - 3 * obs_std,
        obs_mean + 3 * obs_std,
        "data/mean_posterior.csv");

    float vm = thrust_mean(d_v);
    float vs = thrust_std(d_v, vm);
    print_posterior(d_v, "Variance",
        TRUE_SIGMA * TRUE_SIGMA,
        fmaxf(0.0f, vm - 3 * vs),
        vm + 3 * vs,
        "data/variance_posterior.csv");

    printf("Saved:\n");
    write_summary(d_m, d_v);

    float se = obs_std / sqrtf((float)N_OBS);
    printf("\nFrequentist: mean=%.4f +/- %.4f, "
           "95%% CI=[%.4f, %.4f]\n",
           obs_mean, se,
           obs_mean - 1.96f * se,
           obs_mean + 1.96f * se);
}

int main() {
    int grid = (N_RUNS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    long long pred_bytes =
        (long long)N_RUNS * N_PREDICT * sizeof(float);
    mkdir("data", 0755);
    print_header();

    // Observed data
    thrust::device_vector<float> d_obs(N_OBS);
    generate_obs(d_obs);
    float obs_mean = thrust_mean(d_obs);
    float obs_std  = thrust_std(d_obs, obs_mean);
    printf("Observed: mean=%.4f, std=%.4f\n\n",
           obs_mean, obs_std);

    // Allocate cuRAND states, prediction buffer
    curandState_t* d_states;
    float* d_pred;
    cudaMalloc(&d_states,
        N_RUNS * sizeof(curandState_t));
    cudaMalloc(&d_pred, pred_bytes);
    printf("Prediction buffer: %.0f MB\n\n",
           pred_bytes / (1024.0 * 1024.0));

    // Posterior resampling
    thrust::device_vector<float> d_m(N_RUNS);
    thrust::device_vector<float> d_v(N_RUNS);
    run_mean_posterior(
        d_obs, d_pred, d_m, d_states, grid);
    run_var_posterior(
        d_obs, d_pred, d_v, d_states, grid);

    print_results(d_m, d_v, obs_mean, obs_std);

    cudaFree(d_states);
    cudaFree(d_pred);
    return 0;
}
