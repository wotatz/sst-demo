#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;


inline Buffer::SharedPtr asBuffer(Resource::SharedPtr pResource)
{
    return pResource ? std::dynamic_pointer_cast<Buffer>(pResource->shared_from_this()) : nullptr;
}

inline TypedBufferBase::SharedPtr asTypedBuffer(Resource::SharedPtr pResource)
{
    return pResource ? std::dynamic_pointer_cast<TypedBufferBase>(pResource->shared_from_this()) : nullptr;
}

inline StructuredBuffer::SharedPtr asStructuredBuffer(Resource::SharedPtr pResource)
{
    return pResource ? std::dynamic_pointer_cast<StructuredBuffer>(pResource->shared_from_this()) : nullptr;
}

inline Texture::SharedPtr asTexture(Resource::SharedPtr pResource)
{
    return pResource ? std::dynamic_pointer_cast<Texture>(pResource->shared_from_this()) : nullptr;
}

class PassData
{
public:
    template<typename ResourceType>
    void addResource(std::string Name, std::shared_ptr<ResourceType> pResource)
    {
        if (std::is_base_of<Resource, ResourceType>::value)
        {
            mResources[Name] = std::dynamic_pointer_cast<Resource>(pResource);
        }
        else
        {
            logErrorAndExit("PassData: passed variable is not a resource: '" + Name + "'", true);
        }
    }

    bool removeResource(std::string Name);

    const Resource::SharedPtr& operator[](const std::string& Name) const;

    void setWidth(int width)   { mWidth = width; }
    void setHeight(int height) { mHeight = height; }

    ivec2 getExtend() const { return { mWidth, mHeight }; }
    int getWidth()    const { return mWidth; }
    int getHeight()   const { return mHeight; }

    const auto& getResourceMap() const { return mResources; }

    template<typename T>
    T& getVariable(const std::string& Name)
    {
        should_not_get_here();
    }

    template<>
    int& getVariable<int>(const std::string& Name) { return mIntVariables[Name]; }
    template<>
    float& getVariable<float>(const std::string& Name) { return mFloatVariables[Name]; }

private:
    std::unordered_map<std::string, Resource::SharedPtr> mResources;
    std::unordered_map<std::string, int> mIntVariables;
    std::unordered_map<std::string, float> mFloatVariables;
    int mWidth;
    int mHeight;
};
