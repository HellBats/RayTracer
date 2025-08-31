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
    vec3 albedo;
}Sphere;

typedef struct Plane {
    vec3 point;
    vec3 normal;
    vec3 albedo;
}Plane;

typedef struct Triangle {
    TriVertices vertices;
    vec3 normal;
    float origin_distance;
    bool is_double_sided;
    vec3 albedo;
}Triangle;

typedef struct Geometry {
    GeometryType type;
    union {
        Sphere sphere;
        Plane plane;
        Triangle triangle;
    };
}Geometry;

void InitalizeSphere(Sphere &sphere,float &radius, vec3 &center, vec3 &albedo);
void InitalizePlane(Plane &plane,vec3 &point, vec3 &normal, vec3 &albedo);
void InitalizeTriangle(Triangle &triangle,TriVertices &vertices, vec3 &albedo);
__host__ __device__ bool Intersect(Geometry& g,Ray& r, float& t, float &u, float &v);
__host__ __device__ bool IntersectSphere(Sphere &sphere,Ray& r,float& t,float &u, float &v);
__host__ __device__ bool IntersectPlane(Plane &plane ,Ray& r,float &t,float &u, float &v);
__host__ __device__ bool IntersectTriangle(Triangle &traingle,Ray& r,float &t, float &u, float &v);