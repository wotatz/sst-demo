#include "PassData.h"

bool PassData::removeResource(std::string Name)
{
    if (mResources.find(Name) != mResources.end())
    {
        mResources.erase(Name);
        return true;
    }
    return false;
}

const Resource::SharedPtr& PassData::operator[](const std::string& Name) const
{
    static const Resource::SharedPtr pNull;
    return mResources.find(Name) != mResources.end() ? mResources.at(Name) : pNull;
}
