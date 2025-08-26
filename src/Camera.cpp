#include "include/Camera.h"

Camera::Camera(uint32_t image_width,uint32_t image_height,glm::vec3 pos, glm::vec3 rotation)
{
    this->aspect_ratio = (float)image_width/(float)image_height;
    this->fov = 51.2*M_PI/180; // Default FOV, can be adjusted later
}


void Camera::SetCameraPos(glm::vec3 pos)
{
    return;
}
