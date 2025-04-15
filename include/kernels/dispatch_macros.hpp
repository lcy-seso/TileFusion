// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define DISPATCH_TYPE_CASE(TYPE, NV_TYPE, ...) \
    case TYPE: {                               \
        using scalar_t = NV_TYPE;              \
        return __VA_ARGS__();                  \
    }

#define TILEFUSION_DISPATCH_ALL_TYPES(TYPE, ...)                           \
    c10::ScalarType _type = TYPE;                                          \
    [&] {                                                                  \
        switch (_type) {                                                   \
            DISPATCH_TYPE_CASE(c10::ScalarType::Float, float, __VA_ARGS__) \
            DISPATCH_TYPE_CASE(c10::ScalarType::Half, __half, __VA_ARGS__) \
            DISPATCH_TYPE_CASE(c10::ScalarType::BFloat16, __bfloat16,      \
                               __VA_ARGS__)                                \
            default:                                                       \
                AT_ERROR("Dispatch is not implemented for type: '",        \
                         toString(_type), "'");                            \
        }                                                                  \
    }();

#define TILEFUSION_DISPATCH_INTEGER(INTEGER, NAME, ...)               \
    [&] {                                                             \
        switch (INTEGER) {                                            \
            case 64: {                                                \
                static constexpr int NAME = 64;                       \
                return __VA_ARGS__();                                 \
            }                                                         \
            case 128: {                                               \
                static constexpr int NAME = 128;                      \
                return __VA_ARGS__();                                 \
            }                                                         \
            case 256: {                                               \
                static constexpr int NAME = 256;                      \
                return __VA_ARGS__();                                 \
            }                                                         \
            case 512: {                                               \
                static constexpr int NAME = 512;                      \
                return __VA_ARGS__();                                 \
            }                                                         \
            case 1024: {                                              \
                static constexpr int NAME = 1024;                     \
                return __VA_ARGS__();                                 \
            }                                                         \
            default:                                                  \
                AT_ERROR("Dispatch is not implemented for integer: ", \
                         INTEGER);                                    \
        }                                                             \
    }();
