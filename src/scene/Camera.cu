#include "Camera.h"

void InitializeCamera(Camera *camera,uint32_t image_width,uint32_t image_height,vec3 position,vec3 rotation)
{
    camera->image_width = image_width;
    camera->image_height = image_height;
    camera->aspect_ratio = (float)image_width/(float)image_height;
    camera->fov = 51.2*M_PI/180; // Default FOV, can be adjusted later
    camera->position = position;
    camera->rotation = rotation;
    float cx = cosf(camera->rotation.x*M_PI/180); // pitch (x-axis)
    float sx = sinf(camera->rotation.x*M_PI/180);

    float cy = cosf(camera->rotation.y*M_PI/180); // yaw (y-axis)
    float sy = sinf(camera->rotation.y*M_PI/180);

    float cz = cosf(camera->rotation.z*M_PI/180); // roll (z-axis)
    float sz = sinf(camera->rotation.z*M_PI/180);

    // Rotation matrices (column-major convention)
    // Rz * Ry * Rx (roll → yaw → pitch)
    float r00 = cy * cz;
    float r01 = sx * sy * cz - cx * sz;
    float r02 = cx * sy * cz + sx * sz;

    float r10 = cy * sz;
    float r11 = sx * sy * sz + cx * cz;
    float r12 = cx * sy * sz - sx * cz;

    float r20 = -sy;
    float r21 = sx * cy;
    float r22 = cx * cy;

    // Fill transformation (column-major basis vectors + translation)
    camera->transformation.x = vec4{ r00, r10, r20, 0.0f }; // right
    camera->transformation.y = vec4{ r01, r11, r21, 0.0f }; // up
    camera->transformation.z = vec4{ r02, r12, r22, 0.0f }; // forward
    camera->transformation.w = vec4{ camera->position.x,
                                     camera->position.y,
                                     camera->position.z,
                                     1.0f };
}

__host__ __device__ void InitializeTransformation(Camera *camera)
{
    float cx = cosf(camera->rotation.x*M_PI/180); // pitch (x-axis)
    float sx = sinf(camera->rotation.x*M_PI/180);

    float cy = cosf(camera->rotation.y*M_PI/180); // yaw (y-axis)
    float sy = sinf(camera->rotation.y*M_PI/180);

    float cz = cosf(camera->rotation.z*M_PI/180); // roll (z-axis)
    float sz = sinf(camera->rotation.z*M_PI/180);

    // Rotation matrices (column-major convention)
    // Rz * Ry * Rx (roll → yaw → pitch)
    float r00 = cy * cz;
    float r01 = sx * sy * cz - cx * sz;
    float r02 = cx * sy * cz + sx * sz;

    float r10 = cy * sz;
    float r11 = sx * sy * sz + cx * cz;
    float r12 = cx * sy * sz - sx * cz;

    float r20 = -sy;
    float r21 = sx * cy;
    float r22 = cx * cy;

    // Fill transformation (column-major basis vectors + translation)
    camera->transformation.x = vec4{ r00, r10, r20, 0.0f }; // right
    camera->transformation.y = vec4{ r01, r11, r21, 0.0f }; // up
    camera->transformation.z = vec4{ r02, r12, r22, 0.0f }; // forward
    camera->transformation.w = vec4{ camera->position.x,
                                     camera->position.y,
                                     camera->position.z,
                                     1.0f };
}
