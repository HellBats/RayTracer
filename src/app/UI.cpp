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
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar|ImGuiTreeNodeFlags_DefaultOpen);

    // ----------------- Camera Controls -----------------
    if (ImGui::CollapsingHeader("Camera Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::CollapsingHeader("Camera Translations", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Camera X", &(scene.camera.position.x), -40, 40);
            ImGui::SliderFloat("Camera Y", &(scene.camera.position.y), -40, 40);
            ImGui::SliderFloat("Camera Z", &(scene.camera.position.z), -40, 40);
        }

        if (ImGui::CollapsingHeader("Camera Rotations", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Camera Rx", &(scene.camera.rotation.x), -90, 90);
            ImGui::SliderFloat("Camera Ry", &(scene.camera.rotation.y), -90, 90);
            ImGui::SliderFloat("Camera Rz", &(scene.camera.rotation.z), -90, 90);
        }
    }

    // ----------------- Light Controls -----------------
    if (ImGui::CollapsingHeader("Light Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < scene.lights_count; i++)
        {
            ImGui::PushID(i);
            if (ImGui::CollapsingHeader(("Light " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat(("Intensity##" + std::to_string(i)).c_str(), &(scene.lights[i].intensity), 0, 200);

                if (scene.lights[i].type == LightType::DISTANT)
                {
                    ImGui::SliderFloat("Direction X", &(scene.lights[i].distant_light.direction.x), -200, 60);
                    ImGui::SliderFloat("Direction Y", &(scene.lights[i].distant_light.direction.y), -200, 60);
                    ImGui::SliderFloat("Direction Z", &(scene.lights[i].distant_light.direction.z), -200, 60);
                }
                else
                {
                    ImGui::SliderFloat("Position X", &(scene.lights[i].point_light.position.x), -200, 60);
                    ImGui::SliderFloat("Position Y", &(scene.lights[i].point_light.position.y), -200, 60);
                    ImGui::SliderFloat("Position Z", &(scene.lights[i].point_light.position.z), -200, 60);
                }
            }
            ImGui::PopID();
        }
    }

    // ----------------- Object Controls -----------------
    if (ImGui::CollapsingHeader("Object Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < scene.object_count; i++)
        {
            ImGui::PushID(i);
            if (ImGui::CollapsingHeader(("Object " + std::to_string(i)).c_str()))
            {
                ImGui::SliderFloat("Reflectivity", &(scene.objects[i].material.reflectivity), 0, 1);
                ImGui::SliderFloat("Roughness", &(scene.objects[i].material.roughness), 0, 1);
                ImGui::SliderFloat("Refractive Index", &(scene.objects[i].material.refractive_index), 1, 2);
                ImGui::SliderFloat("Transparency", &(scene.objects[i].material.transparency), 0, 1);
            }
            ImGui::PopID();
        }
    }

    // ----------------- Render Controls -----------------
    if (ImGui::Button("Render")) render = !render;
    ImGui::Text("Frame Time: %.2f ms", ms);

    ImGui::End();
}