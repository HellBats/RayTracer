cmake_minimum_required(VERSION 3.10)
project(CPURenderer)
set(CMAKE_CXX_STANDARD 17)

# GLFW
add_subdirectory(glfw)

# GLAD
add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# IMGUI
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
add_library(imgui
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)

# Main executable
add_executable(cpu_renderer src/main.cpp)
target_link_libraries(cpu_renderer PRIVATE glfw glad imgui)
target_include_directories(cpu_renderer PRIVATE external/glad/include)
