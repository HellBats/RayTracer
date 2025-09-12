#include "shader/shader.h"
#include<iostream>


__host__ __device__ float saturate(float x){ return fminf(fmaxf(x, 0.0f), 1.0f); }

__host__ __device__ vec3 Fresnel_Schlick(float cosTheta, vec3 F0) {
    return F0 + (vec3{1.0f,1.0f,1.0f} - F0) * powf(1.0f - cosTheta, 5.0f);
}

__host__ __device__ float Distribution_GGX(float NdotH, float roughness) {
    // a = alpha = roughness^2 (common practice)
    float a = fmaxf(roughness * roughness, 1e-4f);
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PI * denom * denom;
    return a2 / denom;
}

__host__ __device__ float Geometry_Schlick_GGX(float NdotV, float roughness) {
    float r = fmaxf(roughness, 1e-4f);
    float k = (r + 1.0f);
    k = (k * k) / 8.0f;                  // k = (r+1)^2 / 8
    return NdotV / (NdotV * (1.0f - k) + k);
}

__host__ __device__ float Geometry_Smith(float NdotV, float NdotL, float roughness) {
    float ggx1 = Geometry_Schlick_GGX(NdotV, roughness);
    float ggx2 = Geometry_Schlick_GGX(NdotL, roughness);
    return ggx1 * ggx2;
}

__host__ __device__ vec3 CookTorranceBRDF(vec3 N, vec3 V, vec3 L, Material &material)
{
    N = normalize(N);
    V = normalize(V);
    L = normalize(L);

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    // printf("%f %f\n",NdotL,NdotV);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return vec3{0.0f, 0.0f, 0.0f};

    // F0
    vec3 F0;
    if (material.metallic) {
        F0 = F0_conductor(N,
                          material.refractive_index_conductors_k,
                          material.refractive_index_conductors_n);
    } else {
        F0 = F0_dielectric(1.0f, material.refractive_index);
    }
    // F0 = vec3{0.95, 0.64, 0.54};

    vec3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // D, F, G
    float D = Distribution_GGX(NdotH, material.roughness);
    vec3 F = Fresnel_Schlick(VdotH, F0);
    float G = Geometry_Smith(NdotV, NdotL, material.roughness);

    // Cook-Torrance specular
    float denom = 4.0f * NdotV * NdotL + 1e-5f;
    vec3 specular = (D * G) * F * (1.0f / denom);

    // Diffuse (energy-conserving)
    vec3 diffuse = vec3{0.0f, 0.0f, 0.0f};
    if (!material.metallic) {
        // note: use (1 - F) per-channel (F is Fresnel at that H, which approximates directional Fresnel)
        diffuse = (vec3{1.0f,1.0f,1.0f} - F) * material.albedo * (1.0f / M_PI);
    }
    return diffuse+specular;
}



__host__ __device__ void Shade(Light &light, Ray &ray, HitRecord &new_record, HitRecord &old_record,
    vec3 &color)
{
    if(!IsinShadow(light,new_record.t,ray.origin)){
        float light_reflection_intensity = GetLightIntensity(light,old_record.intersection,old_record.normal);
        // Shaded color
        color += CookTorranceBRDF(old_record.normal,-1*ray.direction,
            normalize(GetLightDirection(light,old_record.intersection)),old_record.material)
         * light.color *light_reflection_intensity;
    }
    return;
}