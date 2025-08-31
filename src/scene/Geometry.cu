#include "Geometry.h"
#include<iostream>


const float kepsilon = 1e-6; 

__host__ __device__ bool solveQuadratic(float a, float b,float c,float *t0,float *t1)
{   
    float discriminant  = b*b-4*a*c;
    if(discriminant<0) return false;
    *t0 = (-b + sqrt(discriminant))/(2*a); 
    *t1 = (-b - sqrt(discriminant))/(2*a);
    return true;
}


void InitalizeSphere(Sphere &sphere,float &radius, vec3 &center, vec3 &albedo)
{
    sphere.radius = radius;
    sphere.center = center;
    sphere.albedo = albedo;
}

void InitalizePlane(Plane &plane,vec3 &point, vec3 &normal, vec3 &albedo)
{
    plane.point = point;
    plane.normal = normal;
    plane.albedo = albedo;
}

void InitalizeTriangle(Triangle &triangle,TriVertices &vertices, vec3 &albedo)
{
    triangle.vertices.a = vertices.a;
    triangle.vertices.b = vertices.b;
    triangle.vertices.c = vertices.c;
    triangle.normal = cross((vertices.b-vertices.a),(vertices.c-vertices.a)); 
    triangle.origin_distance = -dot(vertices.a,triangle.normal);
    triangle.is_double_sided = true;
    triangle.albedo = albedo;
}


__host__ __device__ bool Intersect(Geometry& g,Ray& r, float& t, float& u, float& v) {
    switch (g.type) {
        case GeometryType::SPHERE:
            return IntersectSphere(g.sphere, r, t,u,v);
        case GeometryType::PLANE:
            return IntersectPlane(g.plane, r, t,u,v);
        case GeometryType::TRIANGLE:
            return IntersectTriangle(g.triangle, r, t, u, v);
    }
    return false;
}

__host__ __device__ bool IntersectSphere(Sphere &sphere,Ray& r,float& t,float &u, float &v)
{
    vec3 L = r.origin - sphere.center;
    float a = dot(r.direction,r.direction);
    float b = 2 * dot(r.direction,L);
    float c = dot(L,L) - sphere.radius * sphere.radius;
    float t0,t1;
    if (!solveQuadratic(a, b, c, &t0, &t1)) return false;
    // printf("%f, %f\n",t0,t1);
    t = t0>t1?t1:t0;
    if(t>0) 
    {
        u=2;
        v=2;
        return true;
    }
    return false;
}


__host__ __device__ bool IntersectPlane(Plane &plane ,Ray& r,float &t,float &u, float &v)
{
    float denominator = dot(r.direction,plane.normal);
    if(denominator>kepsilon)
    {
        t = dot((plane.point - r.origin),plane.normal)/denominator;
        if(t>=0)
        {
            u=2;
            v=2;
            return true;
        }
    }
    return false;
}

__host__ __device__ bool IntersectTriangle(Triangle &triangle,Ray& r,float& t, float& u, float& v)
{
    
    vec3 v0v1 = triangle.vertices.b - triangle.vertices.a;
    vec3 v0v2 = triangle.vertices.c - triangle.vertices.a;
    vec3 pvec = cross(v0v2,r.direction);
    float det = dot(pvec,v0v1);
    // If the determinant is negative, the triangle is back-facing.
    // If the determinant is close to 0, the ray misses the triangle.
    // If det is close to 0, the ray and triangle are parallel.
    if (fabs(det) < kepsilon) return false;
    float invDet = 1 / det;

    vec3 tvec = r.origin - triangle.vertices.a;
    u = dot(pvec,tvec) * invDet;
    if (u < 0 || u > 1) return false;

    vec3 qvec = cross(v0v1,tvec);
    v = dot(qvec,r.direction) * invDet;
    if (v < 0 || u + v > 1) return false;
    t = dot(qvec,v0v2) * invDet;
    return t>0;
}
