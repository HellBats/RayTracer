#include "include/Ray.h"


Ray::Ray(glm::vec3 origin, glm::vec3 direction) :origin(origin), direction(direction){};
glm::vec3 Ray::GetOrigin(){return origin;};
glm::vec3 Ray::GetDirection(){return direction;};
glm::vec3 Ray::CalculatePoint(float t){return origin+direction*t;};