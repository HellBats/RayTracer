#pragma once
#include "../scene/Scene.h"

__host__ __device__ u8vec3 Shade(Geometry* hitObject, Light* &light, Ray &ray, HitRecord &record, uint32_t light_count);