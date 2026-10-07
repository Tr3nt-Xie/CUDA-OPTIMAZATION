#ifndef LAB6_TIMER_H
#define LAB6_TIMER_H

#include <time.h>

/* C translation units need _POSIX_C_SOURCE for clock_gettime (Makefile). */
static inline double lab6_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#endif
