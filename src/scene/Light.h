#pragma once
#include "../utils/math.h"

struct PointLight
{
    vec3 position;
    vec3 color;
    float intensity;
};


void InitializePointLight(PointLight* light,vec3 position, vec3 color, float intensity);
