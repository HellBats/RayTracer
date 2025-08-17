#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "imgui.h"

class Camera
{
    private:
        float viewport_width;
        float viewport_height;
        float focal_length;
        uint32_t image_width;
        uint32_t image_height;
        glm::vec3 camera_pos;
        glm::vec3 camera_center;
        glm::vec3 x_axis_end;
        glm::vec3 y_axis_end;
        glm::vec3 x_spacing;
        glm::vec3 y_spacing;
        glm::vec3 upper_left_corner; 
        glm::vec3 first_pixel;
    public:
        Camera(float width,float height,float focal_length,glm::vec3 pos,uint32_t image_width,uint32_t image_height);
        void SetCameraPos(glm::vec3 pos);
        float GetWidth();
        float GetHeight();
        float GetFocalLength();
        glm::vec3 GetXSpacing();
        glm::vec3 GetYSpacing();
        glm::vec3 GetCenter();
        glm::vec3 GetFirstPixel();
};