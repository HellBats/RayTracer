#pragma once

#include "utils/math.h"
#include <cuda_runtime.h>

struct Material
{
    vec3 albedo;
    vec3 refractive_index_conductors_n;
    vec3 refractive_index_conductors_k;
    bool metallic;
    float transparency;
    float reflectivity;
    float roughness;
    float refractive_index;
};

__host__ __device__ void InitializeMaterial(Material &material,vec3 albedo,
    vec3 refractive_index_conductors_n,vec3 refractive_index_conductors_k,bool metallic,
    float transparency,float reflectivity,float roughness,float refractive_index);

__host__ __device__ vec3 F0_dielectric(float n_incident, float n_material);
__host__ __device__ vec3 F0_conductor(vec3 n, vec3 k, vec3 n_incident);
__host__ __device__  float fresnel(vec3 I, vec3 N, float ior);