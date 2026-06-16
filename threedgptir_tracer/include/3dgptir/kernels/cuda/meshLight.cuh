// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __CUDACC__

#include <3dgptir/kernels/cuda/sampler.cuh>
#include <3dgptir/mathUtils.h>

static __device__ __forceinline__ bool hasMeshLightData() {
    return params.numMeshLights > 0
        && params.numMeshLightVertices > 0
        && params.numMeshLightTriangles > 0
        && params.meshLightTriangleAliasTable.size(1) >= params.numMeshLightTriangles;
}

static __device__ __forceinline__ bool getMeshLightTriangle(
    const unsigned int triangleId,
    float3& v0,
    float3& v1,
    float3& v2) {
    if (triangleId >= params.numMeshLightTriangles) {
        return false;
    }

    const int i0 = params.meshLightTriangles[triangleId][0];
    const int i1 = params.meshLightTriangles[triangleId][1];
    const int i2 = params.meshLightTriangles[triangleId][2];
    if (i0 < 0 || i1 < 0 || i2 < 0) {
        return false;
    }
    if (static_cast<unsigned int>(i0) >= params.numMeshLightVertices
        || static_cast<unsigned int>(i1) >= params.numMeshLightVertices
        || static_cast<unsigned int>(i2) >= params.numMeshLightVertices) {
        return false;
    }

    v0 = make_float3(
        params.meshLightVertices[i0][0],
        params.meshLightVertices[i0][1],
        params.meshLightVertices[i0][2]);
    v1 = make_float3(
        params.meshLightVertices[i1][0],
        params.meshLightVertices[i1][1],
        params.meshLightVertices[i1][2]);
    v2 = make_float3(
        params.meshLightVertices[i2][0],
        params.meshLightVertices[i2][1],
        params.meshLightVertices[i2][2]);
    return true;
}

static __device__ __forceinline__ bool intersectTriangle(
    const float3& origin,
    const float3& direction,
    const float3& v0,
    const float3& v1,
    const float3& v2,
    float& t,
    float& u,
    float& v) {
    const float3 edge1 = v1 - v0;
    const float3 edge2 = v2 - v0;
    const float3 pvec = cross(direction, edge2);
    const float det = dot(edge1, pvec);
    if (fabsf(det) <= 1e-8f) {
        return false;
    }

    const float invDet = 1.0f / det;
    const float3 tvec = origin - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const float3 qvec = cross(tvec, edge1);
    v = dot(direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    t = dot(edge2, qvec) * invDet;
    return t > 1e-5f;
}

static __device__ __forceinline__ unsigned int sampleMeshTriangleEntry(
    Sampler& sampler,
    const unsigned int triangleOffset,
    const unsigned int triangleCount) {
    const unsigned int localInitial = min(
        static_cast<unsigned int>(sampler.next_1d() * static_cast<float>(triangleCount)),
        triangleCount - 1u);
    const unsigned int initialEntry = triangleOffset + localInitial;
    const unsigned int localAlias = min(
        static_cast<unsigned int>(params.meshLightTriangleAliasTable[1][initialEntry] + 0.5f),
        triangleCount - 1u);
    return sampler.next_1d() < params.meshLightTriangleAliasTable[0][initialEntry]
        ? initialEntry
        : triangleOffset + localAlias;
}

#endif // __CUDACC__
