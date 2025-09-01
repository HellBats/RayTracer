#include "Light.h"



void InitializeLight(Light* light,vec3 position, vec3 color, float intensity)
{
    if(light->type==LightType::POINT) light->point_light.position = position;
    else if(light->type==LightType::DISTANT) light->distant_light.direction = position;
    light->color = color;
    light->intensity = intensity;
}