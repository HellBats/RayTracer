#pragma once
#include "../scene/Scene.h"

#define RAYMAX (50)


struct HitStack
{
    Ray ray[RAYMAX];
    HitRecord records[RAYMAX];
    int ray_count=0;
    int record_count=0;
    Ray* top_ray;
    HitRecord* top_record;
    float bias = 0.1;
    __host__ __device__ void Push(Light* &light, size_t light_count,HitRecord record)
    {
        if(ray_count==RAYMAX-3) return;
        for(int i=0;i<light_count;i++)
        {
            vec3 lightDir;
            float r2;
            Ray r;
            switch (light[i].type)
            {
            case LightType::DISTANT:
                lightDir = normalize(light[i].distant_light.direction*(-1));
                // Lambertian diffuse term
                break;
            
            case LightType::POINT:
                lightDir = normalize(light[i].point_light.position-record.intersection);
                // Lambertian diffuse term
                break;

            default:
                break;
            }
            r = Ray{.origin=record.intersection+record.normal*bias,.direction=lightDir};
            ray[ray_count] = r;
            top_ray = &ray[ray_count];
            ray_count++;
        }
        records[record_count] = record;
        top_record = &records[record_count];
        record_count++;
        // ray[ray_count] = Ray{.origin=record.intersection+record.normal*bias,
        //         .direction=reflect(record.ray_direction,record.normal)};
        // top_ray = &ray[ray_count];
        // ray_count++;
    }

    __host__ __device__ Ray RayPop()
    {
        if(ray_count==0) return Ray{};
        ray_count--;
        return ray[ray_count]; 
    }
    __host__ __device__ Ray RayTop()
    {
        if(ray_count==0) return Ray{};
        return ray[ray_count-1]; 
    }

    __host__ __device__ HitRecord RecordPop()
    {
        if(record_count==0) return HitRecord{};
        record_count--;
        return records[record_count]; 
    }
    __host__ __device__ HitRecord RecordTop()
    {
        if(record_count==0) return HitRecord{};
        return records[record_count-1]; 
    }
    __host__ __device__ bool IsEmpty()
    {
        if(ray_count==0) return true;
        return false; 
    }
};


__host__ __device__ void Shade(Light &light, Ray &ray, HitRecord &new_record, HitRecord &old_record, vec3 &color);
