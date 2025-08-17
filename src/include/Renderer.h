#pragma once
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include "imgui.h"
#include "include/Camera.h"
#include "include/Ray.h"


class Renderer
{
private:
    std::vector<unsigned char>& pixels;
    uint32_t width;
    uint32_t height;
    Camera camera;
public:
    Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height);
    int Render();
    int SetCameraPos(glm::vec3 pos);
};

