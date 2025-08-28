#pragma once
#include "utils/math.h"
#include <glm/glm.hpp>
#include<cuda_runtime.h>

struct Ray
{
    vec3 origin;
    vec3 direction;
};

void IntializeRay(Ray* ray,vec3* origin,vec3* direction);

__host__ __device__ __forceinline__ vec3 CalculatePoint(Ray &ray, float t)
{
    return ray.origin + (ray.direction*t);
};