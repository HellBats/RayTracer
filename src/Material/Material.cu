#include "Material.h"


__host__ __device__ void InitializeMaterial(Material &material,vec3 albedo,
    vec3 refractive_index_conductors_n,vec3 refractive_index_conductors_k,bool metallic,
    float transparency,float reflectivity,float roughness,float refractive_index)
{
    material.albedo = albedo;
    material.transparency = transparency;
    material.metallic = metallic;
    material.reflectivity = reflectivity;
    material.refractive_index_conductors_k = refractive_index_conductors_k;
    material.refractive_index_conductors_n = refractive_index_conductors_n;
    material.roughness = roughness;
    material.refractive_index = refractive_index;
}





__host__ __device__ vec3 F0_dielectric(float n_incident, float n_material) {
    float r = (n_incident - n_material) / (n_incident + n_material);
    float f0 = r * r;
    return vec3{f0, f0, f0};
}

// Conductor F0 (per-channel)
__host__ __device__ vec3 F0_conductor(vec3 n, vec3 k, vec3 n_incident) {
    // component-wise:
    // F0 = ((n - n0)^2 + k^2) / ((n + n0)^2 + k^2)
    vec3 num = (n - n_incident)*(n - n_incident) + k*k;
    vec3 den = (n + n_incident)*(n + n_incident) + k*k;
    return num /den;
}

__host__ __device__  float fresnel(vec3 I, vec3 N, float ior) {
    float cosi = fmaxf(-1.0f, fminf(1.0f, dot(I, N)));
    float etai = 1.0f, etat = ior;
    if (cosi > 0) std::swap(etai, etat);
    float sint = etai/etat * sqrtf(fmaxf(0.f, 1 - cosi*cosi));
    if (sint >= 1) return 1; // total internal reflection
    float cost = sqrtf(fmaxf(0.f, 1 - sint*sint));
    cosi = fabsf(cosi);
    float Rs = ((etat*cosi) - (etai*cost)) / ((etat*cosi) + (etai*cost));
    float Rp = ((etai*cosi) - (etat*cost)) / ((etai*cosi) + (etat*cost));
    return (Rs*Rs + Rp*Rp) / 2.0f;
}