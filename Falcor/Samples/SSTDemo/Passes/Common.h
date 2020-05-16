#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;


template<typename T>
inline std::vector<T> readBuffer(Buffer::SharedPtr buffer)
{
    T* pData = static_cast<T*> (buffer->map(Buffer::MapType::Read));
    size_t numElements = buffer->getSize() / sizeof(T);
    std::vector<T> vec(numElements);
    std::memcpy(vec.data(), pData, numElements * sizeof(T));
    buffer->unmap();
    return vec;
}

template<typename T>
inline void writeBuffer(Buffer::SharedPtr buffer, std::vector<T> dataVec)
{
    T* pData = static_cast<T*> (buffer->map(Buffer::MapType::WriteDiscard));
    size_t numElements = buffer->getSize() / sizeof(T);
    std::vector<T> vec(numElements);
    std::memcpy(pData, vec.data(), numElements * sizeof(T));
    buffer->unmap();
}

template <typename T>
inline T div_round_up(T a, T b) { return (a + b - (T)1) / b; }
