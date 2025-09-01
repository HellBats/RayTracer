#pragma once
#include "scene/Camera.h"
#include "scene/Geometry.h"
#include "scene/Light.h"
#include <memory>


typedef struct Scene {
    Camera camera;
    Light light;
    Geometry* objects;
    uint32_t object_count;

    void push_objects(Geometry &object) {
        objects[object_count] = object;
        object_count++;
    };
    void initializeObjects(size_t size)
    {
        object_count = 0;
        objects = (Geometry*)malloc(sizeof(Geometry)*size);
    }
} Scene;


