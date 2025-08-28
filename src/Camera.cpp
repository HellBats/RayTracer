#include "include/Camera.h"

void InitializeCamera(Camera *camera,uint32_t image_width,uint32_t image_height)
{
    camera->image_width = image_width;
    camera->image_height = image_height;
    camera->aspect_ratio = (float)image_width/(float)image_height;
    camera->fov = 51.2*M_PI/180; // Default FOV, can be adjusted later
}

