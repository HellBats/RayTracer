#pragma once

#include "Material/Material.h"
#include "utils/math.h"


struct HitRecord
{
    float t;
    float u;
    float v;
    vec3 intersection;
    vec3 normal;
    vec3 ray_direction;
    Material material;
    bool front_face;
};
