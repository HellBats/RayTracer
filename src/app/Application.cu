#include "Application.h"

bool HasCUDADevice() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;  // No CUDA devices found
    }
    return true;
}


Application::Application(int window_width, int window_height, int view_width, int view_height)
    : windowWidth(window_width), windowHeight(window_height), viewWidth(view_width), viewHeight(view_height),
      pixels(viewWidth * viewHeight * 4, 255),
      renderer(pixels, viewWidth, viewHeight) 
{
    InitGLFW();
    InitImGui();
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        viewWidth,
        viewHeight,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixels.data()
    );
    bool gpu=HasCUDADevice(); 
    InitializeScene(scene,view_width,view_height,gpu);
    if(gpu)
    {
        Scene *device_scene;
        cudaMalloc(&device_scene, sizeof(Scene));

        // allocate & copy arrays
        Geometry *d_objects;
        Light *d_lights;
        cudaMalloc(&d_objects, sizeof(Geometry) * scene.object_count);
        cudaMalloc(&d_lights, sizeof(Light) * scene.lights_count);

        cudaMemcpy(d_objects, scene.objects,
                sizeof(Geometry) * scene.object_count,
                cudaMemcpyHostToDevice);

        cudaMemcpy(d_lights, scene.lights,
                sizeof(Light) * scene.lights_count,
                cudaMemcpyHostToDevice);

        // patch a copy of Scene on host
        Scene tmp = scene;
        tmp.objects = d_objects;
        tmp.lights = d_lights;

        // copy that fixed struct to GPU
        cudaMemcpy(device_scene, &tmp, sizeof(Scene), cudaMemcpyHostToDevice);

        // now keep device_scene pointer around
        this->device_scene = device_scene;
    }
    
}

Application::~Application() { Cleanup(); }

void Application::Run() {
    unsigned char* device_buffer;
    bool gpu = HasCUDADevice();
    // -------- Allocate GPU buffer --------
    if(gpu) cudaMalloc(&device_buffer, sizeof(unsigned char) * viewWidth * viewHeight * 4); 
    while (!glfwWindowShouldClose(window)) {
        Timer timer;
        glfwPollEvents();
        if (render) {
            // std::fill(pixels.begin(), pixels.end(), 255);
            if (gpu) renderer.RenderGPU(*device_scene,device_buffer);
            else  renderer.RenderCPU(scene);
            // render= !render;
        }

        // Upload pixels to texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                viewWidth, viewHeight,   // not windowWidth/Height
                GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        // UI
        UI::RenderPanels(texture, windowWidth, windowHeight, viewWidth, viewHeight, render,scene,timer.ElapsedMs());
        Scene host_scene = scene;
        Light *device_lights;    // device array for lights
        Geometry *device_objects;

        // allocate device memory for lights/objects
        cudaMalloc(&device_lights, sizeof(Light) * host_scene.lights_count);
        cudaMalloc(&device_objects, sizeof(Geometry) * host_scene.object_count);

        // copy arrays into device arrays
        cudaMemcpy(device_lights, host_scene.lights,
                sizeof(Light) * host_scene.lights_count,
                cudaMemcpyHostToDevice);

        cudaMemcpy(device_objects, host_scene.objects,
                sizeof(Geometry) * host_scene.object_count,
                cudaMemcpyHostToDevice);
        
        host_scene.lights = device_lights;
        host_scene.objects = device_objects;
        cudaMemcpy(device_scene, &host_scene, sizeof(Scene), cudaMemcpyHostToDevice);
        
        // Render
        ImGui::Render();
        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    if(gpu) cudaFree(device_buffer);
}

void Application::InitGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "RayTracer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize GLAD\n";
        exit(EXIT_FAILURE);
    }
}

void Application::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowBorderSize = 0.0f;
    style.ChildBorderSize  = 0.0f;
    style.PopupBorderSize  = 0.0f;
    style.WindowPadding = ImVec2(0,0);
    style.WindowRounding = 0.0f;
}

void Application::Cleanup() {
    cudaFree(scene.objects);
    cudaFree(scene.lights);
    cudaFree(&scene);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}