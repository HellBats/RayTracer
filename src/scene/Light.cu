#include "Light.h"




void InitializeLight(Light* light,vec3 position, vec3 color, float intensity)
{
    if(light->type==LightType::POINT) light->point_light.position = position;
    else if(light->type==LightType::DISTANT) light->distant_light.direction = position;
    light->color = color;
    light->intensity = intensity;
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
        lightDir = normalize(light.point_light.position-point);
        // Lambertian diffuse term
        r2 = norm(lightDir);
        return light.intensity * fmaxf(0.0f, dot(normal, lightDir))/
            (4*M_PI*r2);
        break;

    default:
        return -1;
        break;
    }
    
}