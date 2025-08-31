#include "UI.h"

void UI::RenderPanels(GLuint texture,int window_width, int window_height, int view_width, int view_height
        , bool& render,Scene &scene,float ms) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Image viewport
    ImGui::SetNextWindowPos(ImVec2(0,0));
    ImGui::SetNextWindowSize(ImVec2(view_width, view_height));
    ImGui::Begin("Pixel Buffer", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove);
    ImGui::Image((void*)(intptr_t)texture, ImVec2(view_width, view_height));
    ImGui::End();

    // Control panel
    ImGui::SetNextWindowPos(ImVec2(window_width-300,0));
    ImGui::SetNextWindowSize(ImVec2(300, window_height));
    ImGui::Begin("Controls");
    ImGui::SliderFloat("Camerax",&(scene.camera.position.x),-40,40);
    ImGui::SliderFloat("Cameray",&(scene.camera.position.y),-40,40);
    ImGui::SliderFloat("Cameraz",&(scene.camera.position.z),-40,40);
    ImGui::SliderFloat("CameraRx",&(scene.camera.rotation.x),-90,90);
    ImGui::SliderFloat("CameraRy",&(scene.camera.rotation.y),-90,90);
    ImGui::SliderFloat("CameraRz",&(scene.camera.rotation.z),-90,90);
    ImGui::SliderFloat("LightIntensity",&(scene.light.intensity),0,100);
    ImGui::SliderFloat("lightx",&(scene.light.position.x),-60,60);
    ImGui::SliderFloat("lighty",&(scene.light.position.y),-60,60);
    ImGui::SliderFloat("lightz",&(scene.light.position.z),-200,60);
    if (ImGui::Button("Render")) render = !render;
    ImGui::Text("Frame Time: %.2f ms", ms);
    ImGui::End();
}