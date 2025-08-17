#include "imgui.h"
#include <iostream>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vector>
#include <random>
#include "include/Renderer.h"
#include <chrono>
#include <string>

// Window size
const int WINDOW_WIDTH = 1980;
const int WINDOW_HEIGHT = 1080;

int main()
{
    int view_width = 1080;
    int view_height = 720;
    int side_panel_width = 300;
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Range [min, max]
    int min = 0;
    int max = 255;
    bool render = false;
    // Uniform distribution for integers
    std::uniform_int_distribution<> dist(min, max);
    // ==== Init GLFW ====
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Pixel Buffer Example", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    gladLoadGL();

    // ==== Init ImGui ====
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    // ==== Create CPU Pixel Buffer ====
    std::vector<unsigned char> pixels(view_width * view_height * 4, 255); // RGBA 255=white

    // ==== Create OpenGL Texture ====
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, view_width, view_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    float slider = 400;
    float z = -500; 
    Renderer Scene(pixels,view_width,view_height);
    // ==== Main Loop ====
    while (!glfwWindowShouldClose(window))
    {
        auto start = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        if(render)
        {
            // ==== Fill pixel buffer with Random Noise====
            // for (int y = 0; y < HEIGHT; y++)
            // {
            //     for (int x = 0; x < WIDTH; x++)
            //     {
            //         int idx = (y * WIDTH + x) * 4;
            //         pixels[idx + 0] = dist(gen); // Red
            //         pixels[idx + 1] = dist(gen); // Green
            //         pixels[idx + 2] = dist(gen);                // Blue
            //         pixels[idx + 3] = 255;                // Alpha
            //     }
            // }
            Scene.Render();
        }

        // ==== Upload to GPU ====
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, view_width, view_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        // ==== Start ImGui Frame ====
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Draw the pixel buffer as an image
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(ImVec2(view_width, view_height));
        ImGui::Begin("Pixel Buffer", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Image((void*)(intptr_t)texture, ImVec2(view_width, view_height));
        ImGui::End();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // UI Overlay
        ImGui::SetNextWindowPos(ImVec2(view_width,0));
        ImGui::SetNextWindowSize(ImVec2(300, view_height));
        ImGui::Begin("Controls");
        ImGui::SliderFloat("Radius", &slider, 400, 500);
        ImGui::SliderFloat("CameraZ", &z, -800, 100);
        if (ImGui::Button("Render"))     render=!render;
        char buffer[15];
        sprintf(buffer, "%f", (double)duration.count()/1000);
        ImGui::Text(buffer);
        ImGui::End();

        // ==== Render ====
        ImGui::Render();
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ==== Cleanup ====
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}