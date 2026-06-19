// atlas_timer.h — Cross-platform high-resolution timers
// x86:   __rdtsc() for cycles, QueryPerformanceCounter for wall
// ARM64: cntvct_el0 for cycles, clock_gettime(CLOCK_MONOTONIC) for wall
//
// Usage:
//   #include "atlas_timer.h"
//   double t = atlas_now();          // wall clock, seconds
//   uint64_t c = atlas_cycles();     // CPU cycles (arch-specific counter)
//
// Each TU defines its own profiling macros referencing its own g_prof.
// Example pattern:
//   #define TSC_START()  uint64_t _tsc = atlas_cycles()
//   #define TSC_ACCUM(n) g_prof.n += (atlas_cycles() - _tsc)

#ifndef ATLAS_TIMER_H
#define ATLAS_TIMER_H

#include <cstdint>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <time.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
// 1. Cycle counter
// ═══════════════════════════════════════════════════════════════════════

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #ifdef _MSC_VER
    #include <intrin.h>
    static inline uint64_t atlas_cycles() { return __rdtsc(); }
  #else
    static inline uint64_t atlas_cycles() {
        uint32_t lo, hi;
        __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
  #endif

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
  static inline uint64_t atlas_cycles() {
      uint64_t cnt;
      __asm__ volatile("mrs %0, cntvct_el0" : "=r"(cnt));
      return cnt;
  }

#else
  #warning "atlas_timer.h: no cycle counter for this arch"
  static inline uint64_t atlas_cycles() { return 0; }
#endif

// ═══════════════════════════════════════════════════════════════════════
// 2. Wall-clock: seconds, high-res monotonic
// ═══════════════════════════════════════════════════════════════════════

#ifdef _WIN32
static inline double atlas_now() {
    LARGE_INTEGER c, f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
static inline double atlas_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#endif // ATLAS_TIMER_H
