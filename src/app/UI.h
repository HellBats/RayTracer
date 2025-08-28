#pragma once
#include "imgui.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "scene/Scene.h"

namespace UI {
    void RenderPanels(GLuint texture,int window_width, int window_height, int view_width, int view_height,
         bool& render, Scene &scene,float ms);
}