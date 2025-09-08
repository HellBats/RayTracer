#pragma once
#include "scene/Camera.h"
#include "scene/Geometry.h"
#include "scene/Light.h"
#include <cuda_runtime.h>
#include <memory>


typedef struct Scene {
    Camera camera;
    Light *lights;
    Geometry* objects;
    uint32_t object_count;
    uint32_t lights_count;

    __host__ __device__  void push_objects(Geometry &object) {
        objects[object_count] = object;
        object_count++;
    };
    __host__ __device__  void push_lights(Light &light) {
        lights[lights_count] = light;
        lights_count++;
    };
} Scene;


__host__ __device__ void InitializeScene(Scene &scene, uint32_t viewWidth, uint32_t viewHeight, bool gpu);



