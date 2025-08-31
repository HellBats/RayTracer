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
    __host__ __device__ vec3 operator*(vec3 other) const {
        return vec3{x * other.x, y * other.y, z * other.z};
    }

    __host__ __device__ vec3& operator+=(const vec3 &other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
};


struct vec4 {
    float x, y, z, w;

    // Addition
    __host__ __device__ vec4 operator+(const vec4 &other) const {
        return vec4{x + other.x, y + other.y, z + other.z, w + other.w};
    }

    // Subtraction
    __host__ __device__ vec4 operator-(const vec4 &other) const {
        return vec4{x - other.x, y - other.y, z - other.z, w - other.w};
    }

    // Scalar multiplication
    __host__ __device__ vec4 operator*(float scalar) const {
        return vec4{x * scalar, y * scalar, z * scalar, w * scalar};
    }

    // Compound addition
    __host__ __device__ vec4& operator+=(const vec4 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }
};



struct mat4 {
    vec4 x, y, z, w; // Column Major: 4 columns

    // Operators
    __host__ __device__ mat4 operator+(const mat4 &other) const {
        return mat4{x + other.x, y + other.y, z + other.z, w + other.w};
    }

    __host__ __device__ mat4 operator-(const mat4 &other) const {
        return mat4{x - other.x, y - other.y, z - other.z, w - other.w};
    }

    __host__ __device__ mat4 operator*(float scalar) const {
        return mat4{x * scalar, y * scalar, z * scalar, w * scalar};
    }

    __host__ __device__ mat4& operator+=(const mat4 &other) {
        x += other.x; y += other.y; z += other.z; w += other.w;
        return *this;
    }

    // Multiply matrix by vec4 (mat4 * vec4)
    __host__ __device__ vec4 operator*(const vec4 &v) const {
        // Dot each row with v (remember: stored as columns)
        return vec4{
            x.x * v.x + y.x * v.y + z.x * v.z + w.x * v.w,
            x.y * v.x + y.y * v.y + z.y * v.z + w.y * v.w,
            x.z * v.x + y.z * v.y + z.z * v.z + w.z * v.w,
            x.w * v.x + y.w * v.y + z.w * v.z + w.w * v.w
        };
    }

    // Multiply matrix by vec3 (mat4 * vec3, assuming homogeneous coordinate w=1)
    __host__ __device__ vec3 operator*(const vec3 &v) const {
        float vx = v.x, vy = v.y, vz = v.z;

        float tx = x.x * vx + y.x * vy + z.x * vz + w.x * 1.0f;
        float ty = x.y * vx + y.y * vy + z.y * vz + w.y * 1.0f;
        float tz = x.z * vx + y.z * vy + z.z * vz + w.z * 1.0f;
        float tw = x.w * vx + y.w * vy + z.w * vz + w.w * 1.0f;

        if (tw != 0.0f) {
            tx /= tw;
            ty /= tw;
            tz /= tw;
        }

        return vec3{tx, ty, tz};
    }

    // Multiply mat4 * mat4
    __host__ __device__ mat4 operator*(const mat4 &other) const {
        return mat4{
            (*this) * other.x, // first column
            (*this) * other.y, // second column
            (*this) * other.z, // third column
            (*this) * other.w  // fourth column
        };
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





__host__ __device__ __forceinline__ vec4 intialize(float w, float x, float y, float z) {
    return {w,x, y, z};
}
// Free functions
__host__ __device__ __forceinline__ float dot(const vec4 &a, const vec4 &b) {
    return a.w * b.w+ a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ vec4 normalize(const vec4 &v) {
    float len = sqrtf(v.w*v.w+v.x*v.x + v.y*v.y + v.z*v.z);
    if(len > 0.0f) return v * (1.0f / len);
    return vec4{0,0,0,0};
}


__host__ __device__ __forceinline__ mat4 intialize(vec4 w, vec4 x, vec4 y, vec4 z) {
    return {w,x, y, z};
}

__host__ __device__ __forceinline__ u8vec3 convert_to_u8vec3(vec3 w) {
    return u8vec3{(uint8_t)w.x,(uint8_t)w.y,(uint8_t)w.z};
}


