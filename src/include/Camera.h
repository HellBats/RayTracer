#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui.h"

class Camera
{
    private:
    public:
        float aspect_ratio;
        float fov;
        uint32_t image_width;
        uint32_t image_height;
        Camera(uint32_t image_width,uint32_t image_height,glm::vec3 pos, glm::vec3 rotation);
        void SetCameraPos(glm::vec3 pos);
};