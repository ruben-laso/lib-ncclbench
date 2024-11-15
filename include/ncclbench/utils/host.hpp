#pragma once

#include <unistd.h>

#include <cstdint>

// hashing the host name as it's done in the nccl examples
inline uint64_t get_host_hash(const char *string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

// getting hostname
inline void get_hostname(char *hostname, const size_t maxlen) {
    gethostname(hostname, maxlen);
    for (size_t i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}