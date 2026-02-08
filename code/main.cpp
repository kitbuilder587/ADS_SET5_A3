#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <bitset>

class RandomStreamGen {
public:
    RandomStreamGen(uint64_t seed = 42) : rng_(seed) {}

    std::vector<std::string> generate(size_t n) {
        std::vector<std::string> stream;
        stream.reserve(n);
        std::uniform_int_distribution<int> len_dist(1, 30);
        std::uniform_int_distribution<int> char_dist(0, (int)alphabet_.size() - 1);
        for (size_t i = 0; i < n; ++i) {
            int len = len_dist(rng_);
            std::string s(len, 'x');
            for (int j = 0; j < len; ++j) {
                s[j] = alphabet_[char_dist(rng_)];
            }
            stream.push_back(std::move(s));
        }
        return stream;
    }

    static std::vector<size_t> getPartitions(size_t total, int step_pct) {
        std::vector<size_t> parts;
        for (int pct = step_pct; pct <= 100; pct += step_pct) {
            size_t idx = (size_t)((double)pct / 100.0 * total);
            if (idx > total) idx = total;
            parts.push_back(idx);
        }
        if (parts.empty() || parts.back() != total) {
            parts.push_back(total);
        }
        return parts;
    }

private:
    std::mt19937_64 rng_;
    const std::string alphabet_ =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-";
};

class HashFuncGen {
public:
    HashFuncGen(uint32_t seed = 0) : seed_(seed) {}

    uint32_t hash(const std::string& key) const {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(key.data());
        int len = (int)key.size();
        const int nblocks = len / 4;

        uint32_t h1 = seed_;
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;

        const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data);
        for (int i = 0; i < nblocks; i++) {
            uint32_t k1 = blocks[i];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }

        const uint8_t* tail = data + nblocks * 4;
        uint32_t k1 = 0;
        switch (len & 3) {
            case 3: k1 ^= tail[2] << 16; [[fallthrough]];
            case 2: k1 ^= tail[1] << 8;  [[fallthrough]];
            case 1: k1 ^= tail[0];
                    k1 *= c1;
                    k1 = rotl32(k1, 15);
                    k1 *= c2;
                    h1 ^= k1;
        }

        h1 ^= (uint32_t)len;
        h1 = fmix32(h1);
        return h1;
    }

    static std::vector<HashFuncGen> generateMultiple(int count, uint32_t base_seed = 0) {
        std::vector<HashFuncGen> funcs;
        funcs.reserve(count);
        for (int i = 0; i < count; ++i) {
            funcs.emplace_back(base_seed + (uint32_t)i * 0x9E3779B9u);
        }
        return funcs;
    }

    uint32_t getSeed() const { return seed_; }

private:
    uint32_t seed_;

    static uint32_t rotl32(uint32_t x, int8_t r) {
        return (x << r) | (x >> (32 - r));
    }

    static uint32_t fmix32(uint32_t h) {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
};

class HyperLogLog {
public:
    HyperLogLog(int B, const HashFuncGen& hasher)
        : B_(B), m_(1u << B), hasher_(hasher), registers_(1u << B, 0)
    {
        assert(B >= 4 && B <= 16);
        if (m_ == 16) {
            alpha_ = 0.673;
        } else if (m_ == 32) {
            alpha_ = 0.697;
        } else if (m_ == 64) {
            alpha_ = 0.709;
        } else {
            alpha_ = 0.7213 / (1.0 + 1.079 / (double)m_);
        }
    }

    void add(const std::string& element) {
        uint32_t h = hasher_.hash(element);
        uint32_t idx = h >> (32 - B_);
        uint32_t w = h << B_;
        int rho_val = rho(w);
        if (rho_val > registers_[idx]) {
            registers_[idx] = (uint8_t)rho_val;
        }
    }

    double estimate() const {
        double Z = 0.0;
        for (uint32_t j = 0; j < m_; ++j) {
            Z += std::pow(2.0, -(double)registers_[j]);
        }
        double E = alpha_ * (double)m_ * (double)m_ / Z;

        if (E <= 2.5 * (double)m_) {
            int V = 0;
            for (uint32_t j = 0; j < m_; ++j) {
                if (registers_[j] == 0) V++;
            }
            if (V > 0) {
                E = (double)m_ * std::log((double)m_ / (double)V);
            }
        }
        double two32 = 4294967296.0;
        if (E > two32 / 30.0) {
            E = -two32 * std::log(1.0 - E / two32);
        }

        return E;
    }

    void reset() {
        std::fill(registers_.begin(), registers_.end(), 0);
    }

    int getB() const { return B_; }
    uint32_t getM() const { return m_; }

    size_t memoryUsage() const {
        return registers_.size() * sizeof(uint8_t);
    }

private:
    int B_;
    uint32_t m_;
    double alpha_;
    HashFuncGen hasher_;
    std::vector<uint8_t> registers_;

    int rho(uint32_t w) const {
        int max_rho = 32 - B_ + 1;
        if (w == 0) return max_rho;
        int r = 1;
        while ((w & 0x80000000u) == 0 && r < max_rho) {
            w <<= 1;
            r++;
        }
        return r;
    }
};

class HyperLogLogPlus {
public:
    HyperLogLogPlus(int B, const HashFuncGen& hasher1, const HashFuncGen& hasher2)
        : B_(B), m_(1u << B), hasher1_(hasher1), hasher2_(hasher2),
          registers_(1u << B, 0), use_sparse_(true)
    {
        assert(B >= 4 && B <= 16);
        if (m_ == 16) alpha_ = 0.673;
        else if (m_ == 32) alpha_ = 0.697;
        else if (m_ == 64) alpha_ = 0.709;
        else alpha_ = 0.7213 / (1.0 + 1.079 / (double)m_);

        sparse_threshold_ = m_ * 6;
    }

    void add(const std::string& element) {
        uint32_t h1 = hasher1_.hash(element);
        uint32_t h2 = hasher2_.hash(element);
        uint64_t h = ((uint64_t)h1 << 32) | (uint64_t)h2;

        if (use_sparse_) {
            sparse_set_.insert(h);
            if (sparse_set_.size() > sparse_threshold_) {
                switchToDense();
            }
        } else {
            addToDense(h);
        }
    }

    double estimate() const {
        if (use_sparse_) {
            std::vector<uint8_t> temp_regs(m_, 0);
            for (uint64_t h : sparse_set_) {
                uint32_t idx = (uint32_t)(h >> (64 - B_));
                uint64_t w = h << B_;
                int r = rho64(w);
                if (r > temp_regs[idx]) {
                    temp_regs[idx] = (uint8_t)r;
                }
            }
            return estimateFromRegisters(temp_regs);
        }
        return estimateFromRegisters(registers_);
    }

    void reset() {
        std::fill(registers_.begin(), registers_.end(), 0);
        sparse_set_.clear();
        use_sparse_ = true;
    }

    int getB() const { return B_; }
    uint32_t getM() const { return m_; }

    size_t memoryUsage() const {
        if (use_sparse_) {
            return sparse_set_.size() * 16 + sizeof(sparse_set_);
        }
        return registers_.size() * sizeof(uint8_t);
    }

private:
    int B_;
    uint32_t m_;
    double alpha_;
    HashFuncGen hasher1_, hasher2_;
    std::vector<uint8_t> registers_;
    std::unordered_set<uint64_t> sparse_set_;
    bool use_sparse_;
    size_t sparse_threshold_;

    void switchToDense() {
        for (uint64_t h : sparse_set_) {
            addToDense(h);
        }
        sparse_set_.clear();
        use_sparse_ = false;
    }

    void addToDense(uint64_t h) {
        uint32_t idx = (uint32_t)(h >> (64 - B_));
        uint64_t w = h << B_;
        int r = rho64(w);
        if (r > registers_[idx]) {
            registers_[idx] = (uint8_t)r;
        }
    }

    int rho64(uint64_t w) const {
        int max_rho = 64 - B_ + 1;
        if (w == 0) return max_rho;
        int r = 1;
        while ((w & 0x8000000000000000ULL) == 0 && r < max_rho) {
            w <<= 1;
            r++;
        }
        return r;
    }

    double estimateFromRegisters(const std::vector<uint8_t>& regs) const {
        double Z = 0.0;
        for (uint32_t j = 0; j < m_; ++j) {
            Z += std::pow(2.0, -(double)regs[j]);
        }
        double E = alpha_ * (double)m_ * (double)m_ / Z;

        if (E <= 2.5 * (double)m_) {
            int V = 0;
            for (uint32_t j = 0; j < m_; ++j) {
                if (regs[j] == 0) V++;
            }
            if (V > 0) {
                E = (double)m_ * std::log((double)m_ / (double)V);
            }
        }
        return E;
    }
};

struct ExperimentResult {
    size_t stream_size;
    std::vector<size_t> steps;
    std::vector<size_t> exact_f0;
    std::vector<double> hll_est;
    std::vector<double> hllp_est;
};

ExperimentResult runExperiment(
    const std::vector<std::string>& stream,
    int B,
    const HashFuncGen& hasher,
    const HashFuncGen& hasher2,
    int step_pct = 5)
{
    ExperimentResult result;
    result.stream_size = stream.size();

    auto partitions = RandomStreamGen::getPartitions(stream.size(), step_pct);

    HyperLogLog hll(B, hasher);
    HyperLogLogPlus hllp(B, hasher, hasher2);
    std::unordered_set<std::string> exact_set;

    size_t prev = 0;
    for (size_t part : partitions) {
        for (size_t i = prev; i < part; ++i) {
            hll.add(stream[i]);
            hllp.add(stream[i]);
            exact_set.insert(stream[i]);
        }
        prev = part;

        result.steps.push_back(part);
        result.exact_f0.push_back(exact_set.size());
        result.hll_est.push_back(hll.estimate());
        result.hllp_est.push_back(hllp.estimate());
    }

    return result;
}

void writeCSV(const std::string& filename, const std::vector<ExperimentResult>& results, int B) {
    std::ofstream out(filename);
    out << "stream_id,step,stream_fraction,exact_f0,hll_estimate,hllp_estimate\n";
    for (size_t s = 0; s < results.size(); ++s) {
        const auto& r = results[s];
        for (size_t i = 0; i < r.steps.size(); ++i) {
            double frac = (double)r.steps[i] / (double)r.stream_size;
            out << s << ","
                << r.steps[i] << ","
                << std::fixed << std::setprecision(4) << frac << ","
                << r.exact_f0[i] << ","
                << std::fixed << std::setprecision(2) << r.hll_est[i] << ","
                << std::fixed << std::setprecision(2) << r.hllp_est[i] << "\n";
        }
    }
    out.close();
}

void testRegisterDistribution(int B, const HashFuncGen& hasher, int num_samples = 1000000) {
    uint32_t m = 1u << B;
    std::vector<int> counts(m, 0);
    for (int i = 0; i < num_samples; ++i) {
        std::string s = "reg_" + std::to_string(i);
        uint32_t h = hasher.hash(s);
        uint32_t idx = h >> (32 - B);
        counts[idx]++;
    }

    double expected = (double)num_samples / m;
    double chi2 = 0.0;
    int min_count = counts[0], max_count = counts[0];
    double sum = 0.0, sum_sq = 0.0;
    for (int c : counts) {
        double diff = (double)c - expected;
        chi2 += diff * diff / expected;
        min_count = std::min(min_count, c);
        max_count = std::max(max_count, c);
        sum += c;
        sum_sq += (double)c * c;
    }
    double mean = sum / m;
    double std_dev = std::sqrt(sum_sq / m - mean * mean);

    double critical = 0;
    if (m == 16) critical = 25.0;
    else if (m == 256) critical = 293.25;
    else if (m == 1024) critical = 1098.52;
    else if (m == 16384) critical = 16693.6;

    std::cout << "  B=" << B << " (m=" << m << "): expected=" << std::fixed << std::setprecision(1) << expected
              << " mean=" << mean << " std=" << std_dev
              << " min=" << min_count << " max=" << max_count << std::endl;
    std::cout << "    chi2=" << std::setprecision(2) << chi2 << " (df=" << (m - 1)
              << ", critical~" << critical << ")";
    if (critical > 0 && chi2 < critical) std::cout << " => PASS";
    else if (critical > 0) std::cout << " => FAIL";
    std::cout << std::endl;
}

void testHashUniformity(const HashFuncGen& hasher, int num_buckets = 256, int num_samples = 1000000) {
    std::vector<int> counts(num_buckets, 0);
    for (int i = 0; i < num_samples; ++i) {
        std::string s = "str_" + std::to_string(i);
        uint32_t h = hasher.hash(s);
        counts[h % num_buckets]++;
    }

    double expected = (double)num_samples / num_buckets;
    double chi2 = 0.0;
    for (int c : counts) {
        double diff = (double)c - expected;
        chi2 += diff * diff / expected;
    }
    std::cout << "Hash uniformity test (chi-squared): " << std::fixed << std::setprecision(2) << chi2
              << " (df=" << (num_buckets - 1) << ", critical ~293.25 at alpha=0.05)" << std::endl;
    if (chi2 < 293.25) {
        std::cout << "  => PASS: distribution is uniform" << std::endl;
    } else {
        std::cout << "  => FAIL: distribution is NOT uniform" << std::endl;
    }
}

int main() {
    std::cout << "=== HyperMegaLogLog Pro Max++ ===" << std::endl;
    std::cout << std::endl;

    std::cout << "--- Hash uniformity test ---" << std::endl;
    HashFuncGen test_hasher(42);
    testHashUniformity(test_hasher);
    std::cout << std::endl;

    const int NUM_STREAMS = 20;
    const std::vector<size_t> STREAM_SIZES = {10000, 100000, 500000};
    const std::vector<int> B_VALUES = {4, 8, 10, 14};
    const int STEP_PCT = 5;

    auto hashers = HashFuncGen::generateMultiple(2, 42);
    HashFuncGen& hasher1 = hashers[0];
    HashFuncGen& hasher2 = hashers[1];

    for (int B : B_VALUES) {
        std::cout << "--- Experiments with B=" << B << " (m=" << (1 << B) << ") ---" << std::endl;

        for (size_t stream_size : STREAM_SIZES) {
            std::vector<ExperimentResult> all_results;

            for (int s = 0; s < NUM_STREAMS; ++s) {
                RandomStreamGen gen(s * 1000 + 777);
                auto stream = gen.generate(stream_size);
                auto result = runExperiment(stream, B, hasher1, hasher2, STEP_PCT);
                all_results.push_back(result);
            }

            std::string csv_name = "data_B" + std::to_string(B) + "_N" + std::to_string(stream_size) + ".csv";
            writeCSV(csv_name, all_results, B);
            std::cout << "  Written " << csv_name << " (" << NUM_STREAMS << " streams x " << stream_size << " elements)" << std::endl;

            double sum_ratio = 0.0, sum_ratio_sq = 0.0;
            double sum_ratio_p = 0.0, sum_ratio_p_sq = 0.0;
            for (const auto& r : all_results) {
                size_t last = r.steps.size() - 1;
                double ratio = r.hll_est[last] / (double)r.exact_f0[last];
                double ratio_p = r.hllp_est[last] / (double)r.exact_f0[last];
                sum_ratio += ratio;
                sum_ratio_sq += ratio * ratio;
                sum_ratio_p += ratio_p;
                sum_ratio_p_sq += ratio_p * ratio_p;
            }
            double mean_ratio = sum_ratio / NUM_STREAMS;
            double std_ratio = std::sqrt(sum_ratio_sq / NUM_STREAMS - mean_ratio * mean_ratio);
            double mean_ratio_p = sum_ratio_p / NUM_STREAMS;
            double std_ratio_p = std::sqrt(sum_ratio_p_sq / NUM_STREAMS - mean_ratio_p * mean_ratio_p);

            double theoretical_std = 1.04 / std::sqrt((double)(1 << B));
            double theoretical_upper = 1.30 / std::sqrt((double)(1 << B));

            std::cout << "    Stream size=" << stream_size
                      << ": HLL mean_ratio=" << std::fixed << std::setprecision(4) << mean_ratio
                      << " std=" << std_ratio
                      << " (theory: " << theoretical_std << " / " << theoretical_upper << ")"
                      << std::endl;
            std::cout << "    Stream size=" << stream_size
                      << ": HLL+ mean_ratio=" << std::fixed << std::setprecision(4) << mean_ratio_p
                      << " std=" << std_ratio_p
                      << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "--- Register distribution analysis (1M unique strings) ---" << std::endl;
    for (int B : B_VALUES) {
        testRegisterDistribution(B, hasher1, 1000000);
    }
    std::cout << std::endl;

    std::cout << "--- Memory comparison ---" << std::endl;
    for (int B : B_VALUES) {
        HyperLogLog hll(B, hasher1);
        HyperLogLogPlus hllp(B, hasher1, hasher2);
        std::cout << "  B=" << B << " (m=" << (1 << B) << "): HLL = " << hll.memoryUsage() << " bytes"
                  << ", HLL++ sparse (empty) = " << hllp.memoryUsage() << " bytes" << std::endl;
    }

    {
        RandomStreamGen gen(42);
        auto stream = gen.generate(100000);
        for (int B : {10, 14}) {
            HyperLogLogPlus hllp(B, hasher1, hasher2);
            for (const auto& s : stream) hllp.add(s);
            std::cout << "  B=" << B << " after 100K elements: HLL++ = " << hllp.memoryUsage() << " bytes" << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "Done! CSV files written to current directory." << std::endl;
    return 0;
}
