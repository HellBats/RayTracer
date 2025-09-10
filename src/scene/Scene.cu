#include "scene/Scene.h"


__host__ __device__ void initializeObjectsAndLights(Scene &scene,size_t objects_capacity,
     size_t lights_capacity)
{
    scene.object_count = 0;
    scene.lights_count = 0;
    scene.objects = (Geometry*)malloc(sizeof(Geometry)*objects_capacity);
    scene.lights = (Light*)malloc(sizeof(Light)*lights_capacity);
    
}


__host__ __device__ void InitializeScene(Scene &scene, uint32_t viewWidth, uint32_t viewHeight, bool gpu)
{
    InitializeCamera(&scene.camera, viewWidth, viewHeight, vec3{10,0,0}, vec3{0,0,0});
    Light light1,light2,light3;
    light1.type = LightType::DISTANT;
    light2.type = LightType::POINT;
    light3.type = LightType::POINT;
    InitializeLight(&light1,vec3{0,0,-35},{0.8,0.8,0.8},5,20);
    InitializeLight(&light2,vec3{0,30,-60},{1,1,1},300,20);
    InitializeLight(&light3,vec3{0,30,-40},{1,1,1},300,20);
    initializeObjectsAndLights(scene,10,5);
    scene.push_lights(light1);
    scene.push_lights(light2);
    scene.push_lights(light3);
    vec3 albedo_sphere = vec3{0.6,0.2,0.2};
    vec3 albedo_triangle = vec3{0.5,0.5,0.5};
    float reflectivity = 0.1;
    float roughness = 1;
    float refractive_index = 1.1;
    float transparency = 0.9;
    vec3 refractive_index_conductors_n = vec3{0.14f, 0.37f, 1.54f}; // gold (n)
    vec3 refractive_index_conductors_k = vec3{3.1f, 2.7f, 1.9f};   // gold (k)
    Geometry sphere,triangle1,triangle2;
    Material ball,plane;
    InitializeMaterial(ball,albedo_sphere,refractive_index_conductors_n,refractive_index_conductors_k,
    false,transparency,reflectivity,roughness,refractive_index);
    InitializeMaterial(plane,albedo_triangle,refractive_index_conductors_n,refractive_index_conductors_k,
    false,0,reflectivity,roughness,refractive_index);
    sphere.type = GeometryType::SPHERE;
    sphere.sphere.radius = 10;
    sphere.sphere.center = vec3{0,0,-60};
    sphere.material = ball;
    triangle1.material = plane;
    triangle2.material = plane;
    triangle1.type = GeometryType::TRIANGLE;
    triangle2.type = GeometryType::TRIANGLE;
    TriVertices tri1, tri2;
    float x = -30,y=-11,z=-20,size_x = 60;
    tri1.a = vec3{x,y,z};
    tri1.b = vec3{x+size_x,y,z};
    tri1.c = vec3{x,y,z-size_x};
    tri2.a = vec3{x+size_x,y,z};
    tri2.b = vec3{x+size_x,y,z-size_x};
    tri2.c = vec3{x,y,z-size_x};
    InitalizeTriangle(triangle1.triangle, tri1);
    InitalizeTriangle(triangle2.triangle, tri2);
    scene.push_objects(triangle1);
    scene.push_objects(triangle2);
    scene.push_objects(sphere);
}