#include "shader/shader.h"



__host__ __device__ u8vec3 Shade(Geometry* hitObject, Light* &light, Ray &ray, HitRecord &record,uint32_t light_count)
{
    if (hitObject)   
    {
        vec3 hitColor{0,0,0};
        vec3 point,normal;
        for(int i=0;i<light_count;i++)
        {
            float light_reflection_intensity;
            switch (hitObject->type)
            {
            case GeometryType::SPHERE:
                // Intersection point
                point  = CalculatePoint(ray, record.t);
                // Surface normal (sphere)
                normal = normalize(point - hitObject->sphere.center);
                // Correct light direction: from surface → light
                light_reflection_intensity = GetLightIntensity(light[i],point,normal);
                break;
            
            case GeometryType::TRIANGLE:
                // Intersection point
                point  = CalculatePoint(ray, record.t);
                // Surface normal (sphere)
                normal = normalize(hitObject->triangle.normal);

                // Correct light direction: from surface → light
                light_reflection_intensity = GetLightIntensity(light[i],point,normal);
                break;
            
            default:
                break;
            }
            // Shaded color
            hitColor += hitObject->material.albedo * light[i].color *
                                light_reflection_intensity * (1.0f / M_PI);
        }
        // Normalize to [0,1] if needed
        float maxVal = fmaxf(hitColor.x, fmaxf(hitColor.y, hitColor.z));
        if (maxVal > 1.0f) hitColor = hitColor*(1.0f / maxVal);

        return convert_to_u8vec3(hitColor * 255.0f);
    }
    return u8vec3{255,255,255};
}