#include "Light.h"
#include<iostream>

__host__ __device__ void InitializeLight(Light* light,vec3 position, vec3 color, float intensity, float intensity_multiplier)
{
    if(light->type==LightType::POINT) light->point_light.position = position;
    else if(light->type==LightType::DISTANT) light->distant_light.direction = position;
    light->color = color;
    light->intensity = intensity;
    light->intensity_multiplier = intensity_multiplier;
}

__host__ __device__ float GetLightIntensity(Light &light,vec3 point, vec3 normal)
{
    vec3 lightDir;
    float r2;
    switch (light.type)
    {
    case LightType::DISTANT:
        lightDir = normalize(light.distant_light.direction*(-1));
        // Lambertian diffuse term
        return light.intensity * fmaxf(0.0f, dot(normal, lightDir));
        break;
    
    case LightType::POINT:
        lightDir = light.point_light.position-point;
        // Lambertian diffuse term
        r2 = norm(lightDir);
        return light.intensity * fmaxf(0.0f, dot(normal, normalize(lightDir)))/
            (4*M_PI*r2)*light.intensity_multiplier;
        break;

    default:
        return -1;
        break;
    }
    
}


__host__ __device__ vec3 GetLightDirection(Light &light,vec3 point)
{
    vec3 lightDir;
    float r2;
    switch (light.type)
    {
    case LightType::DISTANT:
        lightDir = normalize(light.distant_light.direction*(-1));
        break;
    
    case LightType::POINT:
        lightDir = light.point_light.position-point;
        break;

    default:
        break;
    }
    return lightDir;
}

__host__ __device__ bool IsinShadow(Light &light,float &distance_to_intersection, vec3 origin_point)
{
    switch (light.type)
    {
    case LightType::DISTANT:
        if(distance_to_intersection==std::numeric_limits<float>::max()) return false;
        return true;
        break;
    
    case LightType::POINT:
        if(distance_to_intersection<norm(origin_point-light.point_light.position)) return true;
        // printf("d");
        return false;
        break;

    default:
        return -1;
        break;
    }
}