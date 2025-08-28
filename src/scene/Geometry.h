#pragma once
#include "Ray.h"
#include <memory>
#include <cuda_runtime.h>


typedef struct TriVertices {
    vec3 a, b, c;
}TriVertices;

enum class GeometryType { SPHERE, PLANE, TRIANGLE };

typedef struct Sphere {
    vec3 center;
    float radius;
}Sphere;

typedef struct Plane {
    vec3 point;
    vec3 normal;
}Plane;

typedef struct Triangle {
    TriVertices vertices;
    vec3 normal;
    float origin_distance;
    bool is_double_sided;
}Triangle;

typedef struct Geometry {
    GeometryType type;
    union {
        Sphere sphere;
        Plane plane;
        Triangle triangle;
    };
}Geometry;

void InitalizeSphere(Sphere &sphere,float &radius, vec3 &center);
void InitalizePlane(Plane &plane,vec3 &point, vec3 &normal);
void InitalizeTriangle(Triangle &triangle,TriVertices &vertices);
__host__ __device__ bool Intersect(Geometry& g,Ray& r, float& t);
__host__ __device__ bool IntersectSphere(Sphere &sphere,Ray& r,float& t);
__host__ __device__ bool IntersectPlane(Plane &plane ,Ray& r,float &t);
__host__ __device__ bool IntersectTriangle(Triangle &traingle,Ray& r,float &t);