#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <time.h>

// Hand-unrolled with AVX2 while preserving Python's floating-point operation order.
static inline double calculate() noexcept {
    constexpr uint32_t blocks = 50'000'000u; // 200,000,000 iterations / 4

    const __m256d ones = _mm256_set1_pd(1.0);
    const __m256d inc  = _mm256_set1_pd(16.0);

    // Memory order after store: [3,5,7,9] and [11,13,15,17]
    __m256d d0 = _mm256_set_pd(9.0, 7.0, 5.0, 3.0);
    __m256d d1 = _mm256_set_pd(17.0, 15.0, 13.0, 11.0);

    alignas(32) double r[8];
    double result = 1.0;

    for (uint32_t b = 0; b < blocks; ++b) {
        const __m256d q0 = _mm256_div_pd(ones, d0);
        const __m256d q1 = _mm256_div_pd(ones, d1);

        _mm256_store_pd(r, q0);
        _mm256_store_pd(r + 4, q1);

        result -= r[0]; result += r[1];
        result -= r[2]; result += r[3];
        result -= r[4]; result += r[5];
        result -= r[6]; result += r[7];

        d0 = _mm256_add_pd(d0, inc);
        d1 = _mm256_add_pd(d1, inc);
    }

    return result * 4.0;
}

int main() {
    timespec start{}, end{};
    clock_gettime(CLOCK_REALTIME, &start);

    const double result = calculate();

    clock_gettime(CLOCK_REALTIME, &end);

    long long sec = static_cast<long long>(end.tv_sec) - static_cast<long long>(start.tv_sec);
    long long nsec = static_cast<long long>(end.tv_nsec) - static_cast<long long>(start.tv_nsec);
    if (nsec < 0) {
        --sec;
        nsec += 1'000'000'000LL;
    }
    const double elapsed = static_cast<double>(sec) + static_cast<double>(nsec) * 1e-9;

    std::printf("Result: %.12f\n", result);
    std::printf("Execution Time: %.6f seconds\n", elapsed);
    return 0;
}