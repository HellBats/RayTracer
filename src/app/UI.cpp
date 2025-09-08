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
    // ImGui::SetNextWindowPos(ImVec2(window_width-300,0));
    ImGui::SetNextWindowSize(ImVec2(300, window_height));
    ImGui::Begin("Controls");
    if(ImGui::CollapsingHeader("Camera Conrols"))
    {
        if(ImGui::CollapsingHeader("Camera Translations"))
        {
            ImGui::SliderFloat("Camerax",&(scene.camera.position.x),-40,40);
            ImGui::SliderFloat("Cameray",&(scene.camera.position.y),-40,40);
            ImGui::SliderFloat("Cameraz",&(scene.camera.position.z),-40,40);
        }
        if(ImGui::CollapsingHeader("Camera Rotations"))
        {
            ImGui::SliderFloat("CameraRx",&(scene.camera.rotation.x),-90,90);
            ImGui::SliderFloat("CameraRy",&(scene.camera.rotation.y),-90,90);
            ImGui::SliderFloat("CameraRz",&(scene.camera.rotation.z),-90,90);
        }
    }
    if(ImGui::CollapsingHeader("Light Controls"))
        {
            for(int i=0;i<scene.lights_count;i++)
            {
                ImGui::PushID(i);
                if(ImGui::CollapsingHeader("Light"))
                {
                    ImGui::SliderFloat(("Intensity##" + std::to_string(i)).c_str(),&(scene.lights[i].intensity),0,2000);
                    if(scene.lights[0].type==LightType::DISTANT)
                    {
                        ImGui::SliderFloat("directionx",&(scene.lights[i].distant_light.direction.x),-200,60);
                        ImGui::SliderFloat("directiony",&(scene.lights[i].distant_light.direction.y),-200,60);
                        ImGui::SliderFloat("directionz",&(scene.lights[i].distant_light.direction.z),-200,60);
                    }
                    else
                    {
                        ImGui::SliderFloat("positionx",&(scene.lights[i].point_light.position.x),-200,60);
                        ImGui::SliderFloat("positiony",&(scene.lights[i].point_light.position.y),-200,60);
                        ImGui::SliderFloat("positionz",&(scene.lights[i].point_light.position.z),-200,60);
                    }
                }
                ImGui::PopID();
            }
        }
    if (ImGui::Button("Render")) render = !render;
    ImGui::Text("Frame Time: %.2f ms", ms);
    ImGui::End();
}