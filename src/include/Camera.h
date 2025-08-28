#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui.h"

struct Camera
{
    float aspect_ratio;
    float fov;
    uint32_t image_width;
    uint32_t image_height;
};

void InitializeCamera(Camera *camera,uint32_t image_width,uint32_t image_height);