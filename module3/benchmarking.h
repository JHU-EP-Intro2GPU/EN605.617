#ifndef INCLUDED_BENCHMARKING_H
#define INCLUDED_BENCHMARKING_H

#include <chrono>

static std::chrono::time_point<std::chrono::high_resolution_clock> start;
static std::chrono::time_point<std::chrono::high_resolution_clock> stop;

// Start the timer
inline void TIC() {start = std::chrono::high_resolution_clock::now();}

// Call this like this ex. TOC<std::chrono::microseconds>() to get the time precision you want
template<typename T>
inline auto TOC() -> decltype(std::chrono::duration_cast<T> (stop - start).count()) {
    stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<T> (stop - start).count();
}

#endif
