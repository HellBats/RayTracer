#include "UI.h"

void UI::RenderPanels(GLuint texture,int window_width, int window_height, int view_width, int view_height
        , bool& render,Scene &scene,float ms) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Image viewport
    ImGui::SetNextWindowSize(ImVec2(view_width, view_height));
    ImGui::Begin("Pixel Buffer", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove);
    ImGui::Image((void*)(intptr_t)texture, ImVec2(view_width, view_height));
    ImGui::End();

    // Control panel
    ImGui::SetNextWindowPos(ImVec2(window_width-300,0));
    ImGui::SetNextWindowSize(ImVec2(300, window_height));
    ImGui::Begin("Controls");
    ImGui::SliderFloat("Camerax",&(scene.camera.position.x),-20,20);
    ImGui::SliderFloat("CameraRx",&(scene.camera.rotation.x),-90,90);
    if (ImGui::Button("Render")) render = !render;
    ImGui::Text("Frame Time: %.2f ms", ms);
    ImGui::End();
}