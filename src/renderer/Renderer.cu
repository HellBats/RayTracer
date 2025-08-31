#include "Renderer.h"
#include <iostream>

// ------------------ Forward declarations ------------------
__global__ void RenderKernel(Scene* scene,unsigned char* device_buffer, uint32_t width, uint32_t height);
__host__ __device__ bool Trace(Scene* scene,Ray &r, float* tNear,Geometry*& hitObject,float& u, float& v);

// ----------------------------------------------------------

Renderer::Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height)
:pixels(pixels), width(width), height(height)
{

}

__host__ __device__ u8vec3 Color(Geometry* hitObject, PointLight &light, Ray &ray, float &t, float &u , float &v)
{
    if (hitObject)   
    {
        if (hitObject->type == GeometryType::SPHERE)
        {
            // Intersection point
            vec3 point  = CalculatePoint(ray, t);
            // Surface normal (sphere)
            vec3 normal = normalize(point - hitObject->sphere.center);

            // Correct light direction: from surface → light
            vec3 lightDir = normalize(light.position-point);

            // Lambertian diffuse term
            float light_reflection_intensity = light.intensity * fmaxf(0.0f, dot(normal, lightDir));

            // Shaded color
            vec3 hitColor = hitObject->sphere.albedo * light.color *
                            light_reflection_intensity * (1.0f / M_PI);

            // Normalize to [0,1] if needed
            float maxVal = fmaxf(hitColor.x, fmaxf(hitColor.y, hitColor.z));
            if (maxVal > 1.0f) hitColor = hitColor*(1.0f / maxVal);

            return convert_to_u8vec3(hitColor * 255.0f);
        }
    }
    return u8vec3{255,255,255};
}

__host__ __device__ void RenderPixel(Scene* scene, uint32_t i, uint32_t j,
                                     u8vec3 &color, int width, int height)
{
    float scale = tan(scene->camera.fov * 0.5f);
    float Px = (2 * ((i + 0.5f) / width) - 1) * scale * scene->camera.aspect_ratio;
    float Py = (1 - 2 * ((j + 0.5f) / height)) * scale;
    float t,u=2,v=2;
    Geometry* hitObject = nullptr;

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
    Ray ray{rayOriginWorld, rayDirWorld};
    Trace(scene, ray, &t, hitObject,u ,v);
    color = Color(hitObject,scene->light,ray, t,u,v);
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

void Renderer::RenderGPU(Scene &scene)
{
    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    // -------- Allocate GPU buffer --------
    unsigned char* device_buffer;
    cudaMalloc(&device_buffer, sizeof(unsigned char) * width * height * 4);

    // -------- Copy objects --------
    Geometry* d_objects;
    cudaMalloc(&d_objects, sizeof(Geometry) * scene.object_count);
    cudaMemcpy(d_objects, scene.objects,
               sizeof(Geometry) * scene.object_count,
               cudaMemcpyHostToDevice);

    // -------- Prepare patched Scene --------
    Scene scene_copy = scene;        // copy original
    scene_copy.objects = d_objects;  // patch objects pointer

    // -------- Copy Scene to device --------
    Scene* d_scene;
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMemcpy(d_scene, &scene_copy, sizeof(Scene), cudaMemcpyHostToDevice);
    // printf("Host scene objects = %d\n", scene.object_count);
    // -------- Launch kernel --------
    RenderKernel<<<grid, block>>>(d_scene, device_buffer, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // -------- Copy back pixels --------
    cudaMemcpy(pixels.data(), device_buffer,
               width * height * 4 * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // -------- Cleanup --------
    cudaFree(d_objects);
    cudaFree(d_scene);
    cudaFree(device_buffer);
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

__host__ __device__ bool Trace(Scene* scene,Ray &r, float* tNear,Geometry*& hitObject,float& u, float& v)
{
    *tNear = std::numeric_limits<float>::max();
    for (int i=0;i<scene->object_count;i++) {
        float t = std::numeric_limits<float>::max(); 
        if (Intersect(scene->objects[i],r, t,u,v) && t < *tNear) {
            hitObject = &scene->objects[i];
            *tNear = t;
        }
    }
    return (hitObject != nullptr);
}
