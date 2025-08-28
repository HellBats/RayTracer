#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

struct vec3 {
    float x, y, z;

    // Operators
    __host__ __device__ vec3 operator+(const vec3 &other) const {
        return vec3{x + other.x, y + other.y, z + other.z};
    }

    __host__ __device__ vec3 operator-(const vec3 &other) const {
        return vec3{x - other.x, y - other.y, z - other.z};
    }

    __host__ __device__ vec3 operator*(float scalar) const {
        return vec3{x * scalar, y * scalar, z * scalar};
    }

    __host__ __device__ vec3& operator+=(const vec3 &other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
};



struct u8vec3 {
    uint8_t r, g, b;
    // Operators

};


__host__ __device__ __forceinline__ vec3 intialize(float x, float y, float z) {
    return {x, y, z};
}
// Free functions
__host__ __device__ __forceinline__ float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ vec3 cross(const vec3 &a, const vec3 &b) {
    return vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ __forceinline__ vec3 normalize(const vec3 &v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len > 0.0f) return v * (1.0f / len);
    return vec3{0,0,0};
}
