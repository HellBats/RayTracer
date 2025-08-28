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


void InitalizeSphere(Sphere &sphere,float &radius, vec3 &center)
{
    sphere.radius = radius;
    sphere.center = center;
}

void InitalizePlane(Plane &plane,vec3 &point, vec3 &normal)
{
    plane.point = point;
    plane.normal = normal;
}

void InitalizeTriangle(Triangle &triangle,TriVertices &vertices)
{
    triangle.vertices.a = vertices.a;
    triangle.vertices.b = vertices.b;
    triangle.vertices.c = vertices.c;
    triangle.normal = normalize(cross((vertices.b-vertices.a),(vertices.c-vertices.a))); 
    triangle.origin_distance = -dot(vertices.a,triangle.normal);
    triangle.is_double_sided = true;
}


__host__ __device__ bool Intersect(Geometry& g,Ray& r, float& t) {
    switch (g.type) {
        case GeometryType::SPHERE:
            return IntersectSphere(g.sphere, r, t);
        case GeometryType::PLANE:
            return IntersectPlane(g.plane, r, t);
        case GeometryType::TRIANGLE:
            return IntersectTriangle(g.triangle, r, t);
    }
    return false;
}

__host__ __device__ bool IntersectSphere(Sphere &sphere,Ray& r,float& t)
{
    vec3 L = r.origin - sphere.center;
    float a = dot(r.direction,r.direction);
    float b = 2 * dot(r.direction,L);
    float c = dot(L,L) - sphere.radius * sphere.radius;
    float t0,t1;
    if (!solveQuadratic(a, b, c, &t0, &t1)) return false;
    t = t0>t1?t0:t1;
    if(t>0) return true;
    return false;
}


__host__ __device__ bool IntersectPlane(Plane &plane ,Ray& r,float &t)
{
    float denominator = dot(r.direction,plane.normal);
    if(denominator>kepsilon)
    {
        t = dot((plane.point - r.origin),plane.normal)/denominator;
        return (t>=0);
    }
    return false;
}

__host__ __device__ bool IntersectTriangle(Triangle &triangle,Ray& r,float &t)
{
    float NdotRay = dot(r.direction,triangle.normal);
    // std::cout<<NdotRay<<'\n';
    if(NdotRay>0 && !triangle.is_double_sided) return false;
    if(fabs(NdotRay)>kepsilon)
    {
        t = -(dot(r.origin,triangle.normal)+triangle.origin_distance)/NdotRay;
        if(t>0)
        {
            vec3 Ne;
            vec3 P = CalculatePoint(r,t);
            // Test sidedness of P w.r.t. edge v0v1
            vec3 ap = P - triangle.vertices.a;
            Ne = cross((triangle.vertices.b-triangle.vertices.a),ap);
            if (dot(Ne,triangle.normal) < 0) return false; // P is on the right side
        
            // Test sidedness of P w.r.t. edge v2v1
            vec3 bp = P - triangle.vertices.b;
            Ne = cross((triangle.vertices.c-triangle.vertices.b),bp);
            if (dot(Ne,triangle.normal) < 0) return false; // P is on the right side
        
            // Test sidedness of P w.r.t. edge v2v0
            vec3 cp = P - triangle.vertices.c;
            Ne = cross((triangle.vertices.a-triangle.vertices.c),cp);
            if (dot(Ne,triangle.normal) < 0) return false; // P is on the right side
            return true;
        }
    }
    return false;
}
