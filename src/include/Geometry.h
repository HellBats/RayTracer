#pragma once
#include "include/Ray.h"
#include <memory>


typedef struct TriVertices {
    glm::vec3 a, b, c;
}TriVertices;

enum class GeometryType { Sphere, Plane, Triangle };

typedef struct Sphere {
    glm::vec3 center;
    float radius;
}Sphere;

typedef struct Plane {
    glm::vec3 point;
    glm::vec3 normal;
}Plane;

typedef struct Triangle {
    TriVertices vertices;
    glm::vec3 normal;
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

void InitalizeSphere(Sphere &sphere,float &radius, glm::vec3 &center);
void InitalizePlane(Plane &plane,glm::vec3 &point, glm::vec3 &normal);
void InitalizeTriangle(Triangle &triangle,TriVertices &vertices);
bool Intersect(const Geometry& g,Ray& r, float& t);
bool IntersectSphere(const Sphere &sphere,Ray& r,float& t);
bool IntersectPlane(const Plane &plane ,Ray& r,float &t);
bool IntersectTriangle(const Triangle &traingle,Ray& r,float &t);