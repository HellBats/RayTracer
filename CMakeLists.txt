cmake_minimum_required(VERSION 3.10)
project(RayTracer)
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

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(renderer ${SRC_FILES})
target_link_libraries(renderer PRIVATE glfw glad imgui)
target_include_directories(renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(renderer PRIVATE external/glad/include)
