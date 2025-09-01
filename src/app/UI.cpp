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
    ImGui::BeginChild("Camera",ImVec2(300,250));
        ImGui::BeginChild("Camera Translations",ImVec2(300,100));
        ImGui::SliderFloat("Camerax",&(scene.camera.position.x),-40,40);
        ImGui::SliderFloat("Cameray",&(scene.camera.position.y),-40,40);
        ImGui::SliderFloat("Cameraz",&(scene.camera.position.z),-40,40);
        ImGui::EndChild();

        ImGui::BeginChild("Camera Roatations",ImVec2(300,100));
        ImGui::SliderFloat("CameraRx",&(scene.camera.rotation.x),-90,90);
        ImGui::SliderFloat("CameraRy",&(scene.camera.rotation.y),-90,90);
        ImGui::SliderFloat("CameraRz",&(scene.camera.rotation.z),-90,90);
        ImGui::EndChild();
    ImGui::EndChild();
    ImGui::BeginChild("Lights",ImVec2(300,100));
        ImGui::SliderFloat("LightIntensity",&(scene.light.intensity),0,100);
        if(scene.light.type==LightType::DISTANT)
        {
            ImGui::SliderFloat("positionx",&(scene.light.distant_light.direction.x),-60,60);
            ImGui::SliderFloat("positiony",&(scene.light.distant_light.direction.y),-60,60);
            ImGui::SliderFloat("positionz",&(scene.light.distant_light.direction.z),-200,60);
        }
        else
        {
            ImGui::SliderFloat("directionx",&(scene.light.point_light.position.x),-60,60);
            ImGui::SliderFloat("directiony",&(scene.light.point_light.position.y),-60,60);
            ImGui::SliderFloat("directionz",&(scene.light.point_light.position.z),-200,60);
        }
    ImGui::EndChild();
    if (ImGui::Button("Render")) render = !render;
    ImGui::Text("Frame Time: %.2f ms", ms);
    ImGui::End();
}