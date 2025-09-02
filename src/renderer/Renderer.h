#pragma once
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "imgui.h"
#include "scene/Camera.h"
#include "scene/Ray.h"
#include "scene/Scene.h"
#include "utils/math.h"
#include "utils/HitRecord.h"
#include "shader/shader.h"


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

__host__ __device__  void RenderPixel(Scene* scene,uint32_t i, uint32_t j, glm::u8vec3 &color,int width, int height);
__host__ __device__ u8vec3 Trace(Scene* scene,Ray &r,Geometry*& hitObject,HitRecord &record);


