#pragma once
#include "../scene/Scene.h"

__host__ __device__ void Shade(Light &light, Ray &ray, HitRecord &new_record, HitRecord &old_record,
     vec3 &color);
__host__ __device__ vec3 Fresnel_Schlick(float cosTheta, vec3 F0);
__host__ __device__ vec3 CookTorranceBRDF(vec3 N, vec3 V, vec3 L, Material &material);