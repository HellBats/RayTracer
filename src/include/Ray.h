#pragma once
#include <glm/glm.hpp>

class Ray
{
    private:
        glm::vec3 origin;
        glm::vec3 direction;
    public:
        Ray(glm::vec3 origin, glm::vec3 direction);
        glm::vec3 GetOrigin();
        glm::vec3 GetDirection();
        glm::vec3 CalculatePoint(float t);
};

class PrimaryRay : public Ray
{

};
class ShadowRay : public Ray
{

};
class ReflectionRay : public Ray
{

};
class RefractionRay : public Ray
{

};