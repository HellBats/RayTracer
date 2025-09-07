#include "shader/shader.h"
#include<iostream>

__host__ __device__ void Shade(Light &light, Ray &ray, HitRecord &new_record, HitRecord &old_record,
     vec3 &color)
{
    if(!IsinShadow(light,new_record.t,ray.origin)){
        float light_reflection_intensity = GetLightIntensity(light,old_record.intersection,old_record.normal);
        // Shaded color
        color += old_record.material.albedo * light.color *
                            light_reflection_intensity * (1.0f / M_PI);
    }
    return;
}
