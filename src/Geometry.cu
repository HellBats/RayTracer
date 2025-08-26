#include "include/Geometry.h"
#include<iostream>


const float kepsilon = 1e-6; 

bool solveQuadratic(float a, float b,float c,float *t0,float *t1)
{   
    float discriminant  = b*b-4*a*c;
    if(discriminant<0) return false;
    *t0 = (-b + glm::sqrt(discriminant))/(2*a); 
    *t1 = (-b - glm::sqrt(discriminant))/(2*a);
    return true;
}


void InitalizeSphere(Sphere &sphere,float &radius, glm::vec3 &center)
{
    sphere.radius = radius;
    sphere.center = center;
}

void InitalizePlane(Plane &plane,glm::vec3 &point, glm::vec3 &normal)
{
    plane.point = point;
    plane.normal = normal;
}

void InitalizeTriangle(Triangle &triangle,TriVertices &vertices)
{
    triangle.vertices.a = vertices.a;
    triangle.vertices.b = vertices.b;
    triangle.vertices.c = vertices.c;
    triangle.normal = glm::normalize(glm::cross(vertices.b-vertices.a,vertices.c-vertices.a)); 
    triangle.origin_distance = -glm::dot(vertices.a,triangle.normal);
    triangle.is_double_sided = true;
}


__host__ __device__ bool Intersect(const Geometry& g,Ray& r, float& t) {
    switch (g.type) {
        case GeometryType::Sphere:
            return IntersectSphere(g.sphere, r, t);
        case GeometryType::Plane:
            return IntersectPlane(g.plane, r, t);
        case GeometryType::Triangle:
            return IntersectTriangle(g.triangle, r, t);
    }
    return false;
}

__host__ __device__ bool IntersectSphere(const Sphere &sphere,Ray& r,float& t)
{
    glm::vec3 L = r.GetOrigin() - sphere.center;
    float a = glm::dot(r.GetDirection(),r.GetDirection());
    float b = 2 * glm::dot(r.GetDirection(),L);
    float c = glm::dot(L,L) - sphere.radius * sphere.radius;
    float t0,t1;
    if (!solveQuadratic(a, b, c, &t0, &t1)) return false;
    t = t0>t1?t0:t1;
    if(t>0) return true;
    return false;
}


__host__ __device__ bool IntersectPlane(const Plane &plane ,Ray& r,float &t)
{
    float denominator = glm::dot(r.GetDirection(),plane.normal);
    if(denominator>kepsilon)
    {
        t = glm::dot(plane.point - r.GetOrigin(),plane.normal)/denominator;
        return (t>=0);
    }
    return false;
}

__host__ __device__ bool IntersectTriangle(const Triangle &triangle,Ray& r,float &t)
{
    float NdotRay = glm::dot(r.GetDirection(),triangle.normal);
    // std::cout<<NdotRay<<'\n';
    if(NdotRay>0 && !triangle.is_double_sided) return false;
    if(fabs(NdotRay)>kepsilon)
    {
        t = -(glm::dot(r.GetOrigin(),triangle.normal)+triangle.origin_distance)/NdotRay;
        if(t>0)
        {
            glm::vec3 Ne;
            glm::vec3 P = r.CalculatePoint(t);
            // Test sidedness of P w.r.t. edge v0v1
            glm::vec3 ap = P - triangle.vertices.a;
            Ne = glm::cross(triangle.vertices.b-triangle.vertices.a,ap);
            if (glm::dot(Ne,triangle.normal) < 0) return false; // P is on the right side
        
            // Test sidedness of P w.r.t. edge v2v1
            glm::vec3 bp = P - triangle.vertices.b;
            Ne = glm::cross(triangle.vertices.c-triangle.vertices.b,bp);
            if (glm::dot(Ne,triangle.normal) < 0) return false; // P is on the right side
        
            // Test sidedness of P w.r.t. edge v2v0
            glm::vec3 cp = P - triangle.vertices.c;
            Ne = glm::cross(triangle.vertices.a-triangle.vertices.c,cp);
            if (glm::dot(Ne,triangle.normal) < 0) return false; // P is on the right side
            return true;
        }
    }
    return false;
}
