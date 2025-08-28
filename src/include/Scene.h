#pragma once
#include "include/Camera.h"
#include "include/Geometry.h"
#include <vector>
#include <memory>


typedef struct Scene {
    Camera camera;
    Geometry* objects;
    uint32_t object_count;

    void push_objects(Geometry &object) {
        objects[object_count] = object;
        object_count++;
    };
    void initialize(size_t size)
    {
        objects = (Geometry*)malloc(sizeof(Geometry)*size);
    }
} Scene;


