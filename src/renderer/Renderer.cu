#include "Renderer.h"
#include <iostream>

// ------------------ Forward declarations ------------------
__global__ void RenderKernel(Scene* scene,unsigned char* device_buffer, uint32_t width, uint32_t height);
__host__ __device__ u8vec3 Trace(Scene* scene,Ray &r,Geometry*& hitObject,HitRecord &record);

// ----------------------------------------------------------

Renderer::Renderer(std::vector<unsigned char>& pixels,uint32_t width, uint32_t height)
:pixels(pixels), width(width), height(height)
{

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
    Ray ray{rayOriginWorld, rayDirWorld};
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


    // -------- Copy lights --------
    Light* d_lights;
    cudaMalloc(&d_lights, sizeof(Light) * scene.lights_count);
    cudaMemcpy(d_lights, scene.lights,
               sizeof(Light) * scene.lights_count,
               cudaMemcpyHostToDevice);

    

    // -------- Prepare patched Scene --------
    Scene scene_copy = scene;        // copy original
    scene_copy.objects = d_objects;  // patch objects pointer
    scene_copy.lights = d_lights;   // patch lights pointer

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


__host__ __device__ u8vec3 Trace(Scene* scene,Ray &r)
{
    vec3 background_color = vec3{1,1,1};
    size_t depth = 1;
    HitStack stack;
    Geometry hitObject;
    vec3 color = vec3{0,0,0};
    HitRecord record;
    FillIntersectionRecord(scene,r,record);
    if(record.t==std::numeric_limits<float>::max()) return convert_to_u8vec3(background_color*255);
    stack.Push(scene->lights,scene->lights_count,record);
    int counter = scene->lights_count-1;
    while(!stack.IsEmpty())
    {
        r = stack.RayPop();
        FillIntersectionRecord(scene,r,record);
        HitRecord old_record = stack.RecordTop();
        Shade(scene->lights[counter],r,record,old_record,color);
        counter--;
        // printf("%f\n",old_record.t);
    }
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);
    return convert_to_u8vec3(color*255);
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
