#include "Renderer.h"
#include <iostream>

// ------------------ Forward declarations ------------------
__global__ void RenderKernel(Scene* scene,unsigned char* device_buffer, uint32_t width, uint32_t height);
__host__ __device__ u8vec3 Trace(Scene* scene, Ray r);

// ----------------------------------------------------------


Renderer::Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height)
:pixels(pixels), width(width), height(height)
{

}

__host__ __device__ inline unsigned int seedFromIndices(int x, int y, int depth) {
    // Combine pixel coords, depth, and frame index into one integer seed
    unsigned int s = (x * 1973) ^ (y * 9277) ^ (depth * 26699);
    return s;
}


__host__ __device__ void RenderPixel(Scene* scene, uint32_t i, uint32_t j,
                                     u8vec3 &color, int width, int height)
{
    float scale = tan(scene->camera.fov * 0.5f);
    float Px = (2 * ((i + 0.5f) / width) - 1) * scale * scene->camera.aspect_ratio;
    float Py = (1 - 2 * ((j + 0.5f) / height)) * scale;

    // Recompute transformation (world matrix of the camera)
    InitializeTransformation(&scene->camera);
    // printf("%d",scene->object_count);
    // Ray origin = camera position in world space
    vec3 rayOriginWorld = scene->camera.position;
    // Pixel point in camera space (on near plane z=-1)
    vec4 pixelCam = {Px, Py, -1, 0};  // direction, w=0
    // Rotate into world space using camera transform
    pixelCam = scene->camera.transformation * pixelCam;
    vec3 rayDirWorld = normalize(vec3{pixelCam.x,pixelCam.y,pixelCam.z});
    Ray ray{.type = RayType::PrimaryRay,.origin = rayOriginWorld,.direction = rayDirWorld};
    color = Trace(scene, ray);
}

void Renderer::RenderCPU(Scene &scene)
{
    // std::cout<<pixels.size()<<std::endl;
    for(int j=0;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            u8vec3 colors; 
            RenderPixel(&scene,i,j,colors,width,height);
            int idx = (j*width+i)*4;
            pixels[idx + 0] = colors.r;
            pixels[idx + 1] = colors.g;
            pixels[idx + 2] = colors.b;
            pixels[idx + 3] = 255;
        }
    }
    return ;
}

void Renderer::RenderGPU(Scene &scene,unsigned char* device_buffer)
{
    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    RenderKernel<<<grid, block>>>(&scene, device_buffer, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // -------- Copy back pixels --------
    cudaMemcpy(pixels.data(), device_buffer,
               width * height * 4 * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
}

__global__ void RenderKernel(Scene* scene,unsigned char* device_buffer, uint32_t width, uint32_t height)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;
    u8vec3 colors;
    RenderPixel(scene,x,y,colors,width,height);

    int idx = (y * width + x) * 4;
    device_buffer[idx + 0] = colors.r;
    device_buffer[idx + 1] = colors.g;
    device_buffer[idx + 2] = colors.b;
    device_buffer[idx + 3] = 255;
}


__host__ __device__ vec3 Background(Ray& r) {
    vec3 unit_dir = normalize(r.direction);
    float t = 0.5f * (unit_dir.y + 1.0f); // map y from [-1,1] to [0,1]
    vec3 color = (1.0f - t) * vec3{1.0, 1.0, 1.0} + t * vec3{0.5, 0.7, 1.0};
    return color;
}

__host__ __device__ u8vec3 Trace(Scene* scene, Ray r) {
    const float EPS = 1e-4f;
    const int MAX_DEPTH = 5;
    vec3 color = vec3{0,0,0};
    vec3 throughput = vec3{1,1,1};  // multiplicative weight along the path

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        HitRecord rec;
        FillIntersectionRecord(scene, r, rec);

        // If miss: add background contribution
        if (rec.t == std::numeric_limits<float>::max()) {
            color += throughput * Background(r);
            break;
        }

        vec3 viewDir = normalize(-1*r.direction);
        vec3 local = vec3{0,0,0};

        // Direct lighting (shadow rays)
        for (int i = 0; i < scene->lights_count; ++i) {
            Light &L = scene->lights[i];
            vec3 lightDir = normalize(GetLightDirection(L, rec.intersection));
            Ray shadowRay{RayType::ShadowRay, rec.intersection + rec.normal * EPS, lightDir};
            HitRecord shadowHit;
            FillIntersectionRecord(scene, shadowRay, shadowHit);
            if (shadowHit.t == std::numeric_limits<float>::max()) {
                float Li = GetLightIntensity(L, rec.intersection, rec.normal);
                local += CookTorranceBRDF(rec.normal, viewDir, lightDir, rec.material)
                       * L.color * Li * fmaxf(dot(rec.normal, lightDir), 0.0f);
            }
        }

        // Apply diffuse/albedo contribution
        color += throughput * (1.0f - rec.material.transparency) * local;

        // Fresnel
        float F = fresnel(-1*normalize(r.direction), rec.normal, rec.material.refractive_index);

        // Reflection + refraction contributions
        vec3 reflectDir = normalize(reflect(r.direction, rec.normal));
        vec3 refractDir;
        bool hasRefract = refract(r.direction, rec.normal, rec.material.refractive_index, refractDir);

        if (rec.material.transparency > 0.0f && hasRefract) {
            // Decide next ray probabilistically (path tracing style)
            if (F > 0.0f && F < 1.0f) {
                // Russian roulette between reflection/refraction
                if (randomFloat((u_int32_t)(r.origin.x * 12.9898f + 
                                        r.origin.y * 78.233f + 
                                        r.origin.z * 37.719f)) < F) {
                    r = Ray{RayType::ReflectionRay, rec.intersection + rec.normal * EPS, reflectDir};
                    throughput = throughput* rec.material.reflectivity;
                } else {
                    r = Ray{RayType::RefractionRay, rec.intersection - rec.normal * EPS, normalize(refractDir)};
                    throughput = throughput* rec.material.transparency;
                }
            } else if (F >= 1.0f) {
                // pure reflection
                r = Ray{RayType::ReflectionRay, rec.intersection + rec.normal * EPS, reflectDir};
                throughput = throughput* rec.material.reflectivity;
            } else {
                // pure refraction
                r = Ray{RayType::RefractionRay, rec.intersection - rec.normal * EPS, normalize(refractDir)};
                throughput = throughput* rec.material.transparency;
            }
        } else if (rec.material.reflectivity > 0.0f) {
            r = Ray{RayType::ReflectionRay, rec.intersection + rec.normal * EPS, reflectDir};
            throughput = throughput* rec.material.reflectivity;
        } else {
            break; // opaque + no reflection → stop path
        }
    }

    return convert_to_u8vec3(clampv(color, vec3{0,0,0}, vec3{1,1,1})*255);
}


__host__ __device__ void FillIntersectionRecord(Scene* scene,Ray &r, HitRecord &record)
{
    HitRecord nearest;
    Geometry hitObject;
    record.u=2;
    record.v=2;
    record.ray_direction = r.direction; 
    record.t = std::numeric_limits<float>::max();
    nearest = record;
    for (int i=0;i<scene->object_count;i++) 
    {
        if (Intersect(scene->objects[i],r, nearest) && record.t > nearest.t)
        {
            hitObject = scene->objects[i];
            record = nearest;
            record.material = hitObject.material;
        }
    }
}
