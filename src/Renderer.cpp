#include "include/Renderer.h"

Renderer::Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height)
:pixels(pixels), width(width), height(height)
,camera(2.0f,2.0f*((float)(width)/(float)(height)),1.0f,glm::vec3(0.0f,0.0f,0.0f),width,height)
{
}

glm::u8vec3 Color(Ray r)
{
    // glm::vec3 unit_direction = glm::normalize(r.GetDirection());
    // float a = 0.5f*(unit_direction.y + 1.0f);
    // return glm::u8vec3(255.0f*((1.0f-a) * glm::vec3(1.0, 1.0, 1.0) + a*glm::vec3(0.5, 0.7, 1.0)));
    return glm::u8vec3(255,255,255);
}

int Renderer::Render()
{
    glm::vec3 x_spacing = camera.GetXSpacing();
    glm::vec3 y_spacing = camera.GetYSpacing();
    glm::vec3 center = camera.GetCenter();
    glm::vec3 first_pixel = camera.GetFirstPixel();
    glm::vec3 pixel_center = first_pixel;
    for(int j=0;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            pixel_center = first_pixel+ ((float)i*x_spacing) + ((float)j*y_spacing);
            glm::vec3 ray_direction = pixel_center - center;
            Ray ray(center,ray_direction);
            glm::u8vec3 colors = Color(ray);
            int idx = (j*width+i)*4;
            pixels[idx + 0] = colors.r; // Red
            pixels[idx + 1] = colors.g; // Green
            pixels[idx + 2] = colors.b; // Blue
            pixels[idx + 3] = 255;
        }
    }
    return 0;
}

int Renderer::SetCameraPos(glm::vec3 pos)
{
    camera.SetCameraPos(pos);
    return 0;
}

