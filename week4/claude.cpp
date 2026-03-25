
#include <iostream>
#include <iomanip>
#include <chrono>
#include <immintrin.h>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    const long long iterations = 200000000LL;
    const double param1 = 4.0;
    const double param2 = 1.0;

    // We compute sum = sum_{i=1}^{N} [ -1/(i*4-1) + 1/(i*4+1) ]
    // = sum_{i=1}^{N} [ 1/(4i+1) - 1/(4i-1) ]
    // result = 1.0 + sum
    // final = result * 4

    // Use AVX2 with 4 doubles per vector, process 4 iterations at a time
    double result = 1.0;

    // Process with AVX2: 4 doubles at a time
    // For lanes: i, i+1, i+2, i+3
    // contribution: sum of [1/(4k+1) - 1/(4k-1)] for k=i..i+3

    __m256d vresult = _mm256_setzero_pd();
    __m256d vp1 = _mm256_set1_pd(param1);
    __m256d vp2 = _mm256_set1_pd(param2);
    __m256d vstep = _mm256_set1_pd(4.0 * param1); // step for 4 iterations

    // initial i values: 1,2,3,4
    __m256d vi = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);

    long long i = 1;
    long long limit = iterations - 3;

    for (; i <= limit; i += 4) {
        // j_neg = i*param1 - param2
        __m256d j_neg = _mm256_fmsub_pd(vi, vp1, vp2);
        // j_pos = i*param1 + param2
        __m256d j_pos = _mm256_fmadd_pd(vi, vp1, vp2);

        // contribution: 1/j_pos - 1/j_neg
        __m256d inv_neg = _mm256_div_pd(_mm256_set1_pd(1.0), j_neg);
        __m256d inv_pos = _mm256_div_pd(_mm256_set1_pd(1.0), j_pos);

        vresult = _mm256_add_pd(vresult, _mm256_sub_pd(inv_pos, inv_neg));

        vi = _mm256_add_pd(vi, vstep);
    }

    // Horizontal sum of vresult
    double arr[4];
    _mm256_storeu_pd(arr, vresult);
    result += arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remaining iterations
    for (; i <= iterations; i++) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }

    result *= 4.0;

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << "\n";
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << elapsed << " seconds\n";

    return 0;
}
