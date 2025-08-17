#include "include/Camera.h"

Camera::Camera(float width,float height,float focal_length,glm::vec3 pos,uint32_t image_width,uint32_t image_height)
    :viewport_height(height), viewport_width(width) ,focal_length(focal_length), camera_pos(pos) ,
     image_width(image_width), image_height(image_height)
{
    camera_center = glm::vec3(0.0f,0.0f,0.0f);
    x_axis_end = glm::vec3(viewport_width,0.0f,0.0f);
    y_axis_end = glm::vec3(0.0f,-viewport_height,0.0f);
    x_spacing = x_axis_end/=(image_width);
    y_spacing = y_axis_end/=(image_height);
    upper_left_corner = camera_center - glm::vec3(0.0f,0.0f,focal_length) 
        - x_axis_end/2.0f - y_axis_end/2.0f; 
    first_pixel = upper_left_corner + 0.5f*(x_spacing+y_spacing);
}

void Camera::SetCameraPos(glm::vec3 pos)
{
    camera_pos = pos;
    return;
}

float Camera::GetWidth()   {return viewport_width;}
float Camera::GetHeight()   {return viewport_height;}
float Camera::GetFocalLength()  {return focal_length;}
glm::vec3 Camera::GetXSpacing() {return x_spacing;}
glm::vec3 Camera::GetYSpacing() {return y_spacing;}
glm::vec3 Camera::GetCenter() {return camera_center;}
glm::vec3 Camera::GetFirstPixel() {return first_pixel;}