#pragma once
#include "../utils/math.h"
#include "imgui.h"

struct Camera
{
    float aspect_ratio;
    float fov;
    uint32_t image_width;
    uint32_t image_height;
    vec3 position;
    vec3 rotation;
    mat4 transformation;
};

__host__ __device__ void InitializeCamera(Camera *camera,uint32_t image_width,uint32_t image_height,vec3 position, vec3 rotation);
__host__ __device__ void InitializeTransformation(Camera *camera);