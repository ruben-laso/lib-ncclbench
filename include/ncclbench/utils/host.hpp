#pragma once

#include <unistd.h>

#include <cstdint>

#include <string>

// hashing the host name as it's done in the nccl examples
inline auto get_host_hash(const char *string) -> uint64_t {
    std::hash<std::string> hash_fn;
    return hash_fn(string);
}

inline auto get_host_hash(const std::string &string) -> uint64_t {
    return get_host_hash(string.c_str());
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

inline auto get_hostname() -> std::string {
    std::array<char, 1024> hostname{};
    get_hostname(hostname.data(), 1024);
    return {hostname.data()};
}