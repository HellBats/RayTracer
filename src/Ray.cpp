#include "include/Ray.h"


void IntializeRay(Ray* ray,vec3* origin,vec3* direction)
{
    ray->origin.x = origin->x;
    ray->origin.y = origin->y;
    ray->origin.z = origin->z;
    ray->direction.x = direction->x;
    ray->direction.y = direction->y;
    ray->direction.z = direction->z;

}
