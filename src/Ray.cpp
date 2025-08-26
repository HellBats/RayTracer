#include "include/Ray.h"


Ray::Ray(glm::vec3 origin, glm::vec3 direction)
{
    this->origin.x = origin.x;
    this->origin.y = origin.y;
    this->origin.z = origin.z;
    this->direction.x = direction.x;
    this->direction.y = direction.y;
    this->direction.z = direction.z;
};
glm::vec3 Ray::GetOrigin(){return origin;};
glm::vec3 Ray::GetDirection(){return direction;};
glm::vec3 Ray::CalculatePoint(float t){return origin+direction*t;};