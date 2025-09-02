#pragma once
#include "scene/Camera.h"
#include "scene/Geometry.h"
#include "scene/Light.h"
#include <memory>


typedef struct Scene {
    Camera camera;
    Light *lights;
    Geometry* objects;
    uint32_t object_count;
    uint32_t lights_count;

    void push_objects(Geometry &object) {
        objects[object_count] = object;
        object_count++;
    };
    void push_lights(Light &light) {
        lights[lights_count] = light;
        lights_count++;
    };
    void initializeObjectsAndLights(size_t objects_capacity, size_t lights_capacity)
    {
        object_count = 0;
        lights_count = 0;
        objects = (Geometry*)malloc(sizeof(Geometry)*objects_capacity);
        lights = (Light*)malloc(sizeof(Light)*lights_capacity);
    }
} Scene;


