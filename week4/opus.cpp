


#include <cstdio>
#include <chrono>
#include <immintrin.h>

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int iterations = 200000000;
    const double param1 = 4.0;
    const double param2 = 1.0;

    // Using AVX2: process 4 doubles at a time
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d p1_vec = _mm256_set1_pd(param1);
    __m256d p2_vec = _mm256_set1_pd(param2);
    __m256d one_vec = _mm256_set1_pd(1.0);
    __m256d four_vec = _mm256_set1_pd(4.0);

    // Process 4 iterations at a time
    // For i = 1,2,3,4 then 5,6,7,8 etc.
    __m256d base = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);

    int i;
    for (i = 1; i + 3 <= iterations; i += 4) {
        __m256d i_vec = _mm256_add_pd(_mm256_set1_pd((double)(i - 1)), base);
        
        // j_minus = i * param1 - param2
        __m256d j_minus = _mm256_fmsub_pd(i_vec, p1_vec, p2_vec);
        // j_plus = i * param1 + param2
        __m256d j_plus = _mm256_fmadd_pd(i_vec, p1_vec, p2_vec);
        
        // result -= 1/j_minus, result += 1/j_plus
        // contribution = 1/j_plus - 1/j_minus
        __m256d inv_minus = _mm256_div_pd(one_vec, j_minus);
        __m256d inv_plus = _mm256_div_pd(one_vec, j_plus);
        
        sum_vec = _mm256_add_pd(sum_vec, _mm256_sub_pd(inv_plus, inv_minus));
    }

    // Horizontal sum
    double partial[4];
    _mm256_storeu_pd(partial, sum_vec);
    double result = 1.0 + partial[0] + partial[1] + partial[2] + partial[3];

    // Handle remaining iterations
    for (; i <= iterations; i++) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }

    result *= 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", elapsed);

    return 0;
}
