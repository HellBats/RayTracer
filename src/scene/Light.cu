#include "Light.h"



void InitializePointLight(PointLight* light,vec3 position, vec3 color, float intensity)
{
    light->position = position;
    light->color = color;
    light->intensity = intensity;
}