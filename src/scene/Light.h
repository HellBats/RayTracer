#pragma once
#include "../utils/math.h"
#include "scene/Ray.h"
#include "scene/Geometry.h"


enum class LightType {POINT,DISTANT};

struct PointLight
{
    vec3 position;
};

struct DistantLight
{
    vec3 direction;
};

struct Light
{
    LightType type;
    vec3 color;
    float intensity_multiplier;
    float intensity;
    union{
        PointLight point_light;
        DistantLight distant_light;
    };
};

__host__ __device__  void InitializeLight(Light* light,vec3 position, vec3 color, float intensity,float intensity_multiplier);

__host__ __device__ float GetLightIntensity(Light &light,vec3 point, vec3 normal);
__host__ __device__ bool IsinShadow(Light &light,float &distance_to_intersection,vec3 origin_point);
__host__ __device__ vec3 GetLightDirection(Light &light,vec3 point);

__host__ __device__ __forceinline__ vec3 reflect(const vec3 &incident, const vec3 &normal) {
    vec3 reflected = incident - 2*(dot(incident,normal))*normal;
    return reflected;
}

__host__ __device__ __forceinline__ bool refract(vec3 I, vec3 N, float eta, vec3 &refracted) {
    float cosi = fmaxf(-1.0f, fminf(1.0f, dot(I, N)));
    float etai = 1.0f, etat = eta;
    vec3 n = N;
    if (cosi < 0) { cosi = -cosi; } 
    else { std::swap(etai, etat); n = -1*N; }
    float eta_ratio = etai / etat;
    float k = 1 - eta_ratio*eta_ratio*(1 - cosi*cosi);
    if (k < 0) return false; // total internal reflection
    refracted = eta_ratio * I + (eta_ratio * cosi - sqrtf(k)) * n;
    return true;
}
