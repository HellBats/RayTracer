#pragma once
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "imgui.h"
#include "include/Camera.h"
#include "include/Ray.h"
#include "include/Scene.h"
#include "include/math.h"


class Renderer
{
private:
    std::vector<unsigned char>& pixels;
    uint32_t width;
    uint32_t height;
public:
    Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height);
    void RenderCPU(Scene &scene);
    void RenderGPU(Scene &scene);
};

void RenderPixel(Scene* scene,uint32_t i, uint32_t j, glm::u8vec3 &color,int width, int height);
bool Trace(Scene* scene,Ray &r, float* tNear,Geometry*& hitObject);


