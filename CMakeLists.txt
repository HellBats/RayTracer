cmake_minimum_required(VERSION 3.14) # FetchContent needs >= 3.14
project(RayTracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# --------------------------------------------------------
# Force FetchContent to use external/ instead of _deps/
# --------------------------------------------------------
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/external)

# -------------------------------
# GLFW
# -------------------------------
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# -------------------------------
# GLAD
# -------------------------------
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36  # pick a release tag
)
FetchContent_MakeAvailable(glad)

# -------------------------------
# IMGUI
# -------------------------------
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC glfw)
endif()

# -------------------------------
# Sources
# -------------------------------
file(GLOB_RECURSE SRC_CPP src/*.cpp)
file(GLOB_RECURSE SRC_CU  src/*.cu)
add_executable(renderer ${SRC_CPP} ${SRC_CU})

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Fix CUDA arch warning (adjust to your GPU)
set_target_properties(renderer PROPERTIES CUDA_ARCHITECTURES "75")

# Link libraries
target_link_libraries(renderer
    PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    glfw
    glad
    imgui
)

# Include paths (for your own source files)
target_include_directories(renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
