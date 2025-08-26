#include "include/Renderer.h"
#include <cuda_runtime.h>

// ------------------ Forward declarations ------------------
__global__ void RenderKernel(Scene scene,unsigned char* device_buffer, uint32_t width, uint32_t height);
__host__ __device__ bool Trace(Scene &scene,Ray &r, float* tNear,Geometry* &hitObject);

// ----------------------------------------------------------

Renderer::Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height)
:pixels(pixels), width(width), height(height)
{
}

__host__ __device__ glm::u8vec3 Color(Geometry* hitObject, float &t)
{
    if(hitObject)   
    {
        float range = 100.0f;
        return glm::u8vec3(255.0f * glm::vec3(1.0, t/range<1?t/range:1, t/range<1?t/range:1));
    }
    return glm::u8vec3(255,255,255);
}

__host__ __device__ void RenderPixel(Scene &scene,uint32_t i, uint32_t j, glm::u8vec3 &color,int width,int height)
{
    float scale = tan(scene.camera->fov * 0.5f);
    float Px = (2 * ((i + 0.5f) / width) - 1) * scale * scene.camera->aspect_ratio;
    float Py = (1 - 2 * ((j + 0.5f) / height)) * scale;

    float t;
    Geometry* hitObject = nullptr;

    glm::vec3 rayOrigin = glm::vec3(0, 0, 0);
    glm::vec3 rayOriginWorld = rayOrigin;
    glm::vec3 rayPWorld = glm::vec3(Px, Py, -1);
    glm::vec3 rayDirection = glm::normalize(rayPWorld - rayOriginWorld);

    Ray ray(rayOriginWorld,rayDirection);
    Trace(scene,ray,&t,hitObject);
    color = Color(hitObject,t);
}

void Renderer::RenderCPU(Scene &scene)
{
    for(int j=0;j<height;j++)
    {
        for(int i=0;i<width;i++)
        {
            glm::u8vec3 colors; 
            RenderPixel(scene,i,j,colors,width,height);
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
    std::cerr << "object_count = " << scene.object_count << "\n";
    // -------- Allocate GPU buffer --------
    unsigned char* device_buffer;
    cudaMalloc(&device_buffer, sizeof(unsigned char) * width * height * 4);

    // -------- Copy objects --------
    Geometry* d_objects;
    cudaMalloc(&d_objects, sizeof(Geometry) * scene.object_count);
    cudaMemcpy(d_objects, scene.objects,
               sizeof(Geometry) * scene.object_count,
               cudaMemcpyHostToDevice);

    // -------- Copy camera --------
    Camera* d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMemcpy(d_camera, scene.camera, sizeof(Camera), cudaMemcpyHostToDevice);

    // -------- Prepare patched Scene --------
    Scene scene_copy = scene;        // copy original
    scene_copy.camera = d_camera;    // patch camera pointer
    scene_copy.objects = d_objects;  // patch objects pointer

    // -------- Copy Scene to device --------
    Scene* d_scene;
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMemcpy(d_scene, &scene_copy, sizeof(Scene), cudaMemcpyHostToDevice);

    // -------- Launch kernel --------
    RenderKernel<<<grid, block>>>(*d_scene, device_buffer, width, height);
    cudaDeviceSynchronize();

    // -------- Copy back pixels --------
    cudaMemcpy(pixels.data(), device_buffer,
               width * height * 4 * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // -------- Cleanup --------
    cudaFree(d_objects);
    cudaFree(d_camera);
    cudaFree(d_scene);
    cudaFree(device_buffer);
}

__global__ void RenderKernel(Scene scene,unsigned char* device_buffer, uint32_t width, uint32_t height)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    glm::u8vec3 colors;
    RenderPixel(scene,x,y,colors,width,height);

    int idx = (y * width + x) * 4;
    device_buffer[idx + 0] = colors.r;
    device_buffer[idx + 1] = colors.g;
    device_buffer[idx + 2] = colors.b;
    device_buffer[idx + 3] = 255;
}

__host__ __device__ bool Trace(Scene &scene,Ray &r, float* tNear,Geometry* &hitObject)
{
    *tNear = std::numeric_limits<float>::max();
    for (int i=0;i<scene.object_count;i++) {
        float t = std::numeric_limits<float>::max(); 
        if (Intersect(scene.objects[i],r, t) && t < *tNear) {
            hitObject = &scene.objects[i];
            *tNear = t;
        }
    }
    return (hitObject != nullptr);
}
