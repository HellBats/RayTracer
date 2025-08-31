#pragma once
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "../renderer/Renderer.h"
#include "../scene/Scene.h"
#include "UI.h"
#include "utils/Timer.h"
#include "utils/math.h"
#include "scene/Light.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

class Application {
public:
    Application(int window_width, int window_height, int view_width, int view_height);
    ~Application();
    void Run();

private:
    void InitGLFW();
    void InitImGui();
    void Cleanup();

    GLFWwindow* window;
    int windowWidth, windowHeight;
    int viewWidth,viewHeight;
    GLuint texture;
    std::vector<unsigned char> pixels;

    Renderer renderer;
    Scene scene;
    bool render = false;
};
