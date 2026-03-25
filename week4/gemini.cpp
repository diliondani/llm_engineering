
#include <iostream>
#include <chrono>
#include <iomanip>

double calculate(long long iterations, double param1, double param2) {
    double result = 1.0;
    long long i = 1;
    double di = 1.0;
    
    // Unroll by 4 to expose instruction-level parallelism for the divisions,
    // fully hiding the division latency behind the 4-cycle sequential addition latency.
    for (; i <= iterations - 3; i += 4) {
        // Calculate independent denominators
        double j1_0 = di * param1 - param2;
        double j2_0 = di * param1 + param2;
        double j1_1 = (di + 1.0) * param1 - param2;
        double j2_1 = (di + 1.0) * param1 + param2;
        double j1_2 = (di + 2.0) * param1 - param2;
        double j2_2 = (di + 2.0) * param1 + param2;
        double j1_3 = (di + 3.0) * param1 - param2;
        double j2_3 = (di + 3.0) * param1 + param2;

        // Perform independent divisions.
        // Doing this before accumulation enables the CPU's out-of-order execution 
        // to heavily pipeline the divisions ahead of the accumulation chain.
        double d1_0 = 1.0 / j1_0;
        double d2_0 = 1.0 / j2_0;
        double d1_1 = 1.0 / j1_1;
        double d2_1 = 1.0 / j2_1;
        double d1_2 = 1.0 / j1_2;
        double d2_2 = 1.0 / j2_2;
        double d1_3 = 1.0 / j1_3;
        double d2_3 = 1.0 / j2_3;

        // Strictly sequential accumulation to guarantee bit-exact identical output 
        // to Python's IEEE 754 half-to-even float rounding semantics.
        result -= d1_0;
        result += d2_0;
        result -= d1_1;
        result += d2_1;
        result -= d1_2;
        result += d2_2;
        result -= d1_3;
        result += d2_3;
        
        di += 4.0;
    }
    
    // Remainder loop
    for (; i <= iterations; ++i) {
        double j1 = di * param1 - param2;
        result -= (1.0 / j1);
        double j2 = di * param1 + param2;
        result += (1.0 / j2);
        di += 1.0;
    }
    
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(200'000'000, 4.0, 1.0) * 4.0;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << "\n";
    std::cout << "Execution Time: " << std::setprecision(6) << diff.count() << " seconds\n";
    
    return 0;
}
