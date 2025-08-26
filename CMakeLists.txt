cmake_minimum_required(VERSION 3.10)
project(RayTracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# GLFW
add_subdirectory(glfw)

# GLAD
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)
# No need to force linker language anymore since C is enabled

# IMGUI
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)

# Sources
file(GLOB_RECURSE SRC_CPP src/*.cpp)
file(GLOB_RECURSE SRC_CU  src/*.cu)
add_executable(renderer ${SRC_CPP} ${SRC_CU})

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Fix CUDA arch warning (set to your GPU)
set_target_properties(renderer PROPERTIES CUDA_ARCHITECTURES "75")

# Link
target_link_libraries(renderer
    PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    glfw
    imgui
    glad
)

# Includes
target_include_directories(renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(renderer PRIVATE external/glad/include)

# Separable compilation
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
