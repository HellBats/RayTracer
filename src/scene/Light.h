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
