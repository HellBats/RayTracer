#pragma once
#include <chrono>

class Timer {
public:
    Timer() { Reset(); }

    // Reset the start time
    void Reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    // Elapsed time in milliseconds (float)
    float ElapsedMs() const {
        return std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
    }

    // Elapsed time in microseconds
    float ElapsedUs() const {
        return std::chrono::duration<float, std::micro>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
    }

    // Elapsed time in seconds
    float ElapsedSec() const {
        return std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};