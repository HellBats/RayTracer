#pragma once
#include "../scene/Scene.h"

#define RAYMAX (50)


struct HitStack {
    Ray ray[RAYMAX];
    HitRecord records[RAYMAX];
    int ray_count = 0;
    int record_count = 0;

    __host__ __device__ void PushRay(const Ray& r) {
        if (ray_count == RAYMAX) return;
        ray[ray_count++] = r;
    }

    __host__ __device__ void PushRecord(const HitRecord& record) {
        if (record_count == RAYMAX) return;
        records[record_count++] = record;
    }

    __host__ __device__ Ray RayPop() {
        if (ray_count == 0) return Ray{};
        return ray[--ray_count];
    }

    __host__ __device__ HitRecord RecordPop() {
        if (record_count == 0) return HitRecord{};
        return records[--record_count];
    }

    __host__ __device__ Ray& RayTop() {
        return ray[ray_count - 1];
    }

    __host__ __device__ HitRecord& RecordTop() {
        return records[record_count - 1];
    }

    __host__ __device__ bool RayIsEmpty() const { return ray_count == 0; }
    __host__ __device__ bool RecordIsEmpty() const { return record_count == 0; }
};

__host__ __device__ void Shade(Light &light, Ray &ray, HitRecord &new_record, HitRecord &old_record,
     vec3 &color);
__host__ __device__ vec3 Fresnel_Schlick(float cosTheta, vec3 F0);
__host__ __device__ vec3 CookTorranceBRDF(vec3 N, vec3 V, vec3 L, Material &material);