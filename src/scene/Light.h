#pragma once
#include "../utils/math.h"

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
    float intensity;
    union{
        PointLight point_light;
        DistantLight distant_light;
    };
};

void InitializeLight(Light* light,vec3 position, vec3 color, float intensity);
