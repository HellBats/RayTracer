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

    // Setup scene
    InitializeCamera(&scene.camera, viewWidth, viewHeight, vec3{10,0,0}, vec3{0,0,0});
    InitializePointLight(&scene.light,vec3{70,70,0},{1,0,0},1);
    scene.initializeObjects(10);
    vec3 albedo = vec3{0.18,0.18,0.18};
    Geometry sphere,triangle;
    sphere.type = GeometryType::SPHERE;
    sphere.sphere.radius = 10;
    sphere.sphere.center = vec3{0,0,-60};
    sphere.sphere.albedo = albedo;
    triangle.type = GeometryType::TRIANGLE;
    TriVertices tri;
    tri.a = vec3{0,-1,-10};
    tri.b = vec3{40,-1,-40};
    tri.c = vec3{0,-1,-40};
    InitalizeTriangle(triangle.triangle, tri,albedo);
    // scene.push_objects(triangle);
    scene.push_objects(sphere);
}

Application::~Application() { Cleanup(); }

void Application::Run() {
    while (!glfwWindowShouldClose(window)) {
        Timer timer;
        glfwPollEvents();
        if (render) {
            // std::fill(pixels.begin(), pixels.end(), 255);
            if (HasCUDADevice()) 
                renderer.RenderGPU(scene);
            else 
                renderer.RenderCPU(scene);
            // render= !render;
        }

        // Upload pixels to texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                viewWidth, viewHeight,   // not windowWidth/Height
                GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        // UI
        UI::RenderPanels(texture, windowWidth, windowHeight, viewWidth, viewHeight, render,scene,timer.ElapsedMs());

        // Render
        ImGui::Render();
        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}