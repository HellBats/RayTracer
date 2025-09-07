#pragma once

#include "utils/math.h"

struct Material
{
    vec3 albedo;
    float reflectivity;
    float metallic;
    float roughness;
};