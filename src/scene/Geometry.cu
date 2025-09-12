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


__host__ __device__ void InitalizeSphere(Sphere &sphere,float &radius, vec3 &center)
{
    sphere.radius = radius;
    sphere.center = center;
}
__host__ __device__ void InitalizeTriangle(Triangle &triangle,TriVertices &vertices)
{
    triangle.vertices.a = vertices.a;
    triangle.vertices.b = vertices.b;
    triangle.vertices.c = vertices.c;
    triangle.normal = normalize(cross((vertices.b-vertices.a),(vertices.c-vertices.a))); 
    triangle.origin_distance = -dot(vertices.a,triangle.normal);
    triangle.is_double_sided = true;
}


__host__ __device__ void InitializePlane(TriVertices *vertices,vec3 center,vec3 normal,float x, float y)
{
    normal = normalize(normal);

    // pick any vector not parallel to normal
    vec3 up = {0.0f, 1.0f, 0.0f};
    if (std::fabs(dot(up, normal)) > 0.999f) {
        // normal is parallel to up; pick another reference
        up = {1.0f, 0.0f, 0.0f};
    }

    // compute tangent and bitangent (orthonormal basis)
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = normalize(cross(normal, tangent));

    float hw = x * 0.5f;
    float hh = y * 0.5f;

    // four corners (CCW when looking along normal)
    vec3 c0 = center + tangent * (-hw) + bitangent * (-hh); // bottom-left
    vec3 c1 = center + tangent * ( hw) + bitangent * (-hh); // bottom-right
    vec3 c2 = center + tangent * ( hw) + bitangent * ( hh); // top-right
    vec3 c3 = center + tangent * (-hw) + bitangent * ( hh); // top-left
    vertices[0] = {c0,c1,c3};
    vertices[1] = {c1,c2,c3}; 
}

__host__ __device__ void InitializeCube(TriVertices *vertices,vec3 point,float x, float y,float z)
{
    InitializePlane(vertices,point + vec3{x/2,0,z/2},vec3{0,-1,0},x,z);
    InitializePlane(vertices+2,point + vec3{x,y/2,z/2},vec3{1,0,0},y,z);
    InitializePlane(vertices+4,point + vec3{0,y/2,z/2},vec3{-1,0,0},y,z);
    InitializePlane(vertices+6,point + vec3{x/2,y/2,0},vec3{0,0,-1},x,y);
    InitializePlane(vertices+8,point + vec3{x/2,y/2,z},vec3{0,0,1},x,y);
    InitializePlane(vertices+10,point + vec3{x/2,y,z/2},vec3{0,1,0},x,z);
}




__host__ __device__ bool Intersect(Geometry& g,Ray& r,HitRecord &record) {
    switch (g.type) {
        case GeometryType::SPHERE:
            return IntersectSphere(g.sphere, r, record);
        case GeometryType::TRIANGLE:
            return IntersectTriangle(g.triangle, r, record);
    }
    return false;
}

__host__ __device__ bool IntersectSphere(Sphere &sphere,Ray& r,HitRecord &record)
{
    vec3 L = r.origin - sphere.center;
    float a = dot(r.direction,r.direction);
    float b = 2 * dot(r.direction,L);
    float c = dot(L,L) - sphere.radius * sphere.radius;
    float t0,t1;
    if (!solveQuadratic(a, b, c, &t0, &t1)) return false;
    // printf("%f, %f\n",t0,t1);
    record.t = t0>t1?t1:t0;
    record.normal = normalize(CalculatePoint(r,record.t)-sphere.center);
    record.intersection = CalculatePoint(r,record.t);
    if(record.t>0) 
    {
        record.u=2;
        record.v=2;
        return true;
    }
    return false;
}

__host__ __device__ bool IntersectTriangle(Triangle &triangle,Ray& r,HitRecord &record)
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
    record.u = dot(pvec,tvec) * invDet;
    if (record.u < 0 || record.u > 1) return false;

    vec3 qvec = cross(v0v1,tvec);
    record.v = dot(qvec,r.direction) * invDet;
    if (record.v < 0 || record.u + record.v > 1) return false;
    record.normal = triangle.normal;
    record.t = dot(qvec,v0v2) * invDet;
    record.intersection = CalculatePoint(r,record.t);
    return record.t>0;
}
