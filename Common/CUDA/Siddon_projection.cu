/*-------------------------------------------------------------------------
 *
 * CUDA functions for ray-voxel intersection based projection
 *
 * This file has the necesary fucntiosn to perform X-ray CBCT projection
 * operation given a geaometry, angles and image. It usesthe so-called
 * Jacobs algorithm to compute efficiently the length of the x-rays over
 * voxel space.
 *
 * CODE by       Ander Biguri
 *               Sepideh Hatamikia (arbitrary rotation)
 * ---------------------------------------------------------------------------
 * ---------------------------------------------------------------------------
 * Copyright (c) 2015, University of Bath and CERN- European Organization for
 * Nuclear Research
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------
 *
 * Contact: tigre.toolbox@gmail.com
 * Codes  : https://github.com/CERN/TIGRE
 * ---------------------------------------------------------------------------
 */

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Siddon_projection.hpp"
#include "TIGRE_common.hpp"
#include <math.h>

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("%s \n",msg);\
                mexErrMsgIdAndTxt("Ax:Siddon_projection",cudaGetErrorString(__err));\
        } \
} while (0)
    
    
#define MAXTREADS 1024
#define PROJ_PER_BLOCK 9
#define PIXEL_SIZE_BLOCK 9
    /*GEOMETRY DEFINITION
     *
     *                Detector plane, behind
     *            |-----------------------------|
     *            |                             |
     *            |                             |
     *            |                             |
     *            |                             |
     *            |      +--------+             |
     *            |     /        /|             |
     *   A Z      |    /        / |*D           |
     *   |        |   +--------+  |             |
     *   |        |   |        |  |             |
     *   |        |   |     *O |  +             |
     *    --->y   |   |        | /              |
     *  /         |   |        |/               |
     * V X        |   +--------+                |
     *            |-----------------------------|
     *
     *           *S
     *
     *
     *
     *
     *
     **/
    
    void CreateTexture(const GpuIds& gpuids,const float* imagedata,Geometry geo,cudaArray** d_cuArrTex, cudaTextureObject_t *texImage,bool alloc);

__constant__ Point3D projParamsArrayDev[6*PROJ_PER_BLOCK];  // Dev means it is on device


__global__ void vecAddInPlace(float *a, float *b, unsigned long  n)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (idx < n)
        a[idx] = a[idx] + b[idx];
}

__global__ void kernelPixelDetector( Geometry geo,
        float* detector,
        const int currProjSetNumber,
        const int totalNoOfProjections,
        cudaTextureObject_t tex){
    
    unsigned long long u = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long v = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long projNumber=threadIdx.z;
    unsigned long long rayNumber=blockIdx.z;

    float gelTubeRadius = projParamsArrayDev[6 * projNumber + 5].x; //aux2.x;
    u += (unsigned long long)geo.nVoxelY/2 - gelTubeRadius / geo.dVoxelY;
    
    float minU = geo.nVoxelY/2 - gelTubeRadius / geo.dVoxelY;
    float maxU = geo.nVoxelY/2 + gelTubeRadius / geo.dVoxelY;
    
    if (u>maxU ||v>= geo.nDetecV || projNumber>=PROJ_PER_BLOCK)
        return;
    
#if IS_FOR_MATLAB_TIGRE
    size_t idx =  (size_t)(u * (unsigned long long)geo.nDetecV + v)+ projNumber*(unsigned long long)geo.nDetecV *(unsigned long long)geo.nDetecU ;
#else
    size_t idx =  (size_t)(v * (unsigned long long)geo.nDetecU + u)+ projNumber*(unsigned long long)geo.nDetecV *(unsigned long long)geo.nDetecU ;
#endif
    unsigned long indAlpha = currProjSetNumber*PROJ_PER_BLOCK+projNumber;  // This is the ABSOLUTE projection number in the projection array (for a given GPU)

    if(indAlpha>=totalNoOfProjections)
        return;
    
    Point3D uvOrigin = projParamsArrayDev[6 * projNumber];  // 6*projNumber because we have 6 Point3D values per projection
    Point3D deltaU = projParamsArrayDev[6 * projNumber + 1];
    Point3D deltaV = projParamsArrayDev[6 * projNumber + 2];
    Point3D source = projParamsArrayDev[6 * projNumber + 3];
    Point3D auxParams1 = projParamsArrayDev[6 * projNumber + 4];
    Point3D auxParams2 = projParamsArrayDev[6 * projNumber + 5];

    // float gelTubeRadius = auxParams2.x/geo.dVoxelX;
    gelTubeRadius = gelTubeRadius / geo.dVoxelX;
    float gelTubeRadiusSquare = gelTubeRadius * gelTubeRadius;
    float DSD = auxParams1.x;
    float DSO = auxParams1.y;
    float EPS = auxParams1.z;
    float h = geo.dDetecU * (DSD - DSO) / (EPS - geo.dDetecU);
    float fGrowth = EPS * (DSD + h) / (DSD - DSO + h);

            float pixelU, pixelV;

            pixelU = u;
            pixelV = geo.nDetecV-v-1;

            // float uiHC = 2;
            // float pixelU = u + (uiHC ) / (10*2);
            // float pixelV = geo.nDetecV-v-1;

            Point3D pixel1D;
            pixel1D.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
            pixel1D.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
            pixel1D.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);

            // changing source for each u, v
            // o (0,0,0) esta no canto da imagem, uvorigem eh o canto do detector nessa geometria

            float pixelUs = pixelU - (float)(geo.nDetecU - 1) * 0.5f;
            float pixelVs = pixelV - (float)(geo.nDetecV - 1) * 0.5f; 

            //Point3D deltaUd, deltaVd;
            //deltaUd.x = deltaU.x / geo.dDetecU * fGrowth;
            //deltaUd.y = deltaU.y / geo.dDetecU * fGrowth;
            //deltaUd.z = deltaU.z / geo.dDetecU * fGrowth;

            //deltaVd.x = deltaV.x / geo.dDetecV * fGrowth;
            //deltaVd.y = deltaV.y / geo.dDetecV * fGrowth;
            //deltaVd.z = deltaV.z / geo.dDetecV * fGrowth;

            //source.x += pixelUs * deltaUd.x + pixelVs * deltaVd.x;
            //source.y += pixelUs * deltaUd.y + pixelVs * deltaVd.y;
            //source.z += pixelUs * deltaUd.z + pixelVs * deltaVd.z;
            Point3D newSource;

            newSource.x = source.x + fGrowth * ( __fdividef(pixelUs * deltaU.x, geo.dDetecU) + __fdividef(pixelVs * deltaV.x, geo.dDetecV) );
            newSource.y = source.y + fGrowth * ( __fdividef(pixelUs * deltaU.y, geo.dDetecU) + __fdividef(pixelVs * deltaV.y, geo.dDetecV) );
            newSource.z = source.z + fGrowth * ( __fdividef(pixelUs * deltaU.z, geo.dDetecU) + __fdividef(pixelVs * deltaV.z, geo.dDetecV) );

            ///////
            // Siddon's ray-voxel intersection, optimized as in doi=10.1.1.55.7516
            //////
            // Also called Jacobs algorithms
            Point3D ray;
            // vector of Xray
            ray.x=pixel1D.x - newSource.x;
            ray.y=pixel1D.y - newSource.y;
            ray.z=pixel1D.z - newSource.z;

            float eps=0.000001;
            // ray.x=(abs(ray.x)<eps)? 0 : ray.x;
            // ray.y=(abs(ray.y)<eps)? 0 : ray.y;
            // ray.z=(abs(ray.z)<eps)? 0 : ray.z;

            // C = (geo.sVoxelX/2, geo.sVoxelY/2, P.z) center of the tube with P height
            Point3D C;
            C.x = geo.nVoxelX*0.5f;
            C.y = geo.nVoxelY*0.5f;

            // float distanceToCenterInTheMiddleOfImage = (ray.y*0.5f + newSource.y - C.y) * (ray.y*0.5f + newSource.y - C.y) + (ray.x*0.5f + newSource.x - C.x) * (ray.x*0.5f + newSource.x - C.x);

            // // if ray dont pass thrugh gel tube, return 0
            // if ( distanceToCenterInTheMiddleOfImage > gelTubeRadiusSquare) {
            //     // detector[idx] = 0;
            //     return;
            // }

            Point3D Q;
            Q.x = newSource.x - C.x;
            Q.y = newSource.y - C.y;
            Q.z = 0;
            float a1, a2;

            float aux1 = 2 * Q.x * ray.x + 2 * Q.y * ray.y;
            float aux2 = ray.x * ray.x + ray.y * ray.y + ray.z * ray.z;
            float aux3 = (-gelTubeRadiusSquare + Q.x * Q.x + Q.y * Q.y);

            // no real solutions or only 1 solution
            if (aux1*aux1 - 4 * aux2 * aux3 <= 0) return;

            a1 = (-__fsqrt_rd(aux1*aux1 - 4 * aux2 * aux3) - aux1) / (2*aux2);
            a2 = (__fsqrt_rd(aux1*aux1 - 4 * aux2 * aux3) - aux1) / (2*aux2);

            // points where ray intersects the gel tube
            Point3D Q1, Q2;
            Q1.x = newSource.x + a1 * ray.x;
            Q1.y = newSource.y + a1 * ray.y;
            Q1.z = newSource.z + a1 * ray.z;
            Q2.x = newSource.x + a2 * ray.x;
            Q2.y = newSource.y + a2 * ray.y;
            Q2.z = newSource.z + a2 * ray.z;

            float increment = a1/10;
            float lastRadiusSquare = (Q1.x- C.x)*(Q1.x- C.x) + (Q1.y- C.y)*(Q1.y- C.y);
            int iterationsAfterLastChange = 0;

            for (int i = 0; i < 4000; i++) {
                Q1.x = newSource.x + a1 * ray.x;
                Q1.y = newSource.y + a1 * ray.y;
                Q1.z = newSource.z + a1 * ray.z;

                float currentRadiusSquare = (Q1.x- C.x)*(Q1.x- C.x) + (Q1.y- C.y)*(Q1.y- C.y);
                if ( abs(currentRadiusSquare - gelTubeRadiusSquare) < 1) break;

                if (abs(currentRadiusSquare - gelTubeRadiusSquare) > abs(lastRadiusSquare - gelTubeRadiusSquare)) {
                    if (iterationsAfterLastChange <= 2) { //local minimum
                        a1 = a1 - increment;
                        increment = -increment/10;
                    } else {
                        increment = -increment;
                    }

                    iterationsAfterLastChange = -1;
                }

                a1 = a1 + increment;
                lastRadiusSquare = currentRadiusSquare;
                iterationsAfterLastChange++;
            }

            // This variables are ommited because
            // bx,by,bz ={0,0,0}
            // dx,dy,dz ={1,1,1}
            // compute parameter values for x-ray parametric equation. eq(3-10)
            float axm,aym,azm;
            float axM,ayM,azM;

            // In the paper Nx= number of X planes-> Nvoxel+1
            axm=min(__fdividef(Q1.x-newSource.x,ray.x),__fdividef(Q2.x-newSource.x,ray.x));
            aym=min(__fdividef(Q1.y-newSource.y,ray.y),__fdividef(Q2.y-newSource.y,ray.y));
            azm=min(__fdividef(Q1.z-newSource.z,ray.z),__fdividef(Q2.z-newSource.z,ray.z));
            axM=max(__fdividef(Q1.x-newSource.x,ray.x),__fdividef(Q2.x-newSource.x,ray.x));
            ayM=max(__fdividef(Q1.y-newSource.y,ray.y),__fdividef(Q2.y-newSource.y,ray.y));
            azM=max(__fdividef(Q1.z-newSource.z,ray.z),__fdividef(Q2.z-newSource.z,ray.z));
            
            float am=max(max(axm,aym),azm);
            float aM=min(min(axM,ayM),azM);
            
            // line intersects voxel space ->   am<aM
            if (am>=aM){
                // detector[idx]=0;
                return;
            }
            
            // Compute max/min image INDEX for intersection eq(11-19)
            // Discussion about ternary operator in CUDA: https://stackoverflow.com/questions/7104384/in-cuda-why-is-a-b010-more-efficient-than-an-if-else-version
            float imin,imax,jmin,jmax,kmin,kmax;
            // for X
            if( newSource.x<pixel1D.x){
                imin= (newSource.x+am*ray.x);
                imax=(newSource.x+aM*ray.x);
            }else{
                imax=(newSource.x+am*ray.x);
                imin= (newSource.x+aM*ray.x);
            }
            // for Y
            if( newSource.y<pixel1D.y){
                jmin= (newSource.y+am*ray.y);
                jmax=(newSource.y+aM*ray.y);
            }else{
                jmax=(newSource.y+am*ray.y);
                jmin= (newSource.y+aM*ray.y);
            }
            // for Z
            if( newSource.z<pixel1D.z){
                kmin= (newSource.z+am*ray.z);
                kmax=(newSource.z+aM*ray.z);
            }else{
                kmax=(newSource.z+am*ray.z);
                kmin= (newSource.z+aM*ray.z);
            }
            
            // get intersection point N1. eq(20-21) [(also eq 9-10)]
            float ax,ay,az;
            ax=(newSource.x<pixel1D.x)?  __fdividef(imin-newSource.x,ray.x) :  __fdividef(imax-newSource.x,ray.x);
            ay=(newSource.y<pixel1D.y)?  __fdividef(jmin-newSource.y,ray.y) :  __fdividef(jmax-newSource.y,ray.y);
            az=(newSource.z<pixel1D.z)?  __fdividef(kmin-newSource.z,ray.z) :  __fdividef(kmax-newSource.z,ray.z);
            
            // If its Infinite (i.e. ray is perpendicular to this axis), make sure its positive
            ax=(isinf(ax))? abs(ax) : ax;
            ay=(isinf(ay))? abs(ay) : ay;
            az=(isinf(az))? abs(az) : az;
            
            // get index of first intersection. eq (26) and (19)
            float i,j,k;
            float aminc=min(min(ax,ay),az);
            i=newSource.x+ (aminc+am)*0.5f*ray.x;
            j=newSource.y+ (aminc+am)*0.5f*ray.y;
            k=newSource.z+ (aminc+am)*0.5f*ray.z;
            // Initialize
            float ac=am;
            //eq (28), unit anlges
            float axu,ayu,azu;
            axu=__frcp_rd(abs(ray.x));
            ayu=__frcp_rd(abs(ray.y));
            azu=__frcp_rd(abs(ray.z));
            // eq(29), direction of update
            float iu,ju,ku;
            iu=(newSource.x< pixel1D.x)? 1.0f : -1.0f;
            ju=(newSource.y< pixel1D.y)? 1.0f : -1.0f;
            ku=(newSource.z< pixel1D.z)? 1.0f : -1.0f;
            
            float sum=0.0f;
            float traveledLength;

            // refraction to enter tube
            float rayLength = __fsqrt_rd(ray.x*ray.x + ray.y*ray.y + ray.z*ray.z);
            Point3D v1, P, n, N, v2;

            // incident ray versor
            v1.x = __fdividef(ray.x, rayLength);
            v1.y = __fdividef(ray.y, rayLength);
            v1.z = __fdividef(ray.z, rayLength);
            
            // P = (i, j, k) refraction point
            P.x = Q1.x;
            P.y = Q1.y;
            P.z = Q1.z;

            // n = (P - C) / ||P - C|| normal vector to tube surface
            float nLength = __fsqrt_rd((P.x - C.x)*(P.x - C.x) + (P.y - C.y)*(P.y - C.y));
            n.x = (P.x - C.x) / nLength;
            n.y = (P.y - C.y) / nLength;
            n.z = 0; // (P.z - C.z) / nLength

            float incidenceAngle = acos(v1.x*n.x + v1.y*n.y + v1.z*n.z);
            incidenceAngle = (incidenceAngle > PI_2) ? PI_1 - incidenceAngle : incidenceAngle;
            float refractionAngle = asin(nWater * sin(incidenceAngle) / nGel);

            // N = n x v1 normal to the plane formed by n and v1 and v2; n.z = 0
            N.x = -n.y * v1.z;
            N.y = n.x * v1.z;
            N.z = v1.x * n.y - v1.y * n.x;

            float cos_r = cos(refractionAngle);
            float cos_ri = cos(incidenceAngle - refractionAngle);
            
            aux1 = v1.x*n.y*N.z - v1.y*n.x*N.z + v1.z * (n.x*N.y -n.y*N.x);

            v2.x = v1.x;
            v2.y = v1.y;
            v2.z = v1.z;

            // normalize v2
            float v2Length = __fsqrt_rd(v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
            v2.x = __fdividef(v2.x,  v2Length);
            v2.y = __fdividef(v2.y,  v2Length);
            v2.z = __fdividef(v2.z,  v2Length);

            float t;

            Point3D aux;
            aux.x = P.x - C.x;
            aux.y = P.y - C.y;
            aux.z = 0;

            aux1 = 2 * aux.x * v2.x + 2 * aux.y * v2.y;
            aux2 = v2.x * v2.x + v2.y * v2.y + v2.z * v2.z;
            aux3 = (-gelTubeRadiusSquare + aux.x * aux.x + aux.y * aux.y);

            // no real solutions or only 1 solution
            if (aux1*aux1 - 4 * aux2 * aux3 < 0) return;

            a1 = (-__fsqrt_rd(aux1*aux1 - 4 * aux2 * aux3) - aux1) / (2*aux2);
            a2 = (__fsqrt_rd(aux1*aux1 - 4 * aux2 * aux3) - aux1) / (2*aux2);

            t = abs(a1) > abs(a2) ? a1 : a2;
            // end of refraction
            
            Q2.x = P.x + t * v2.x;
            Q2.y = P.y + t * v2.y;
            Q2.z = P.z + t * v2.z;

            increment = t/10;
            lastRadiusSquare = (Q2.x- C.x)*(Q2.x- C.x) + (Q2.y- C.y)*(Q2.y- C.y);
            iterationsAfterLastChange = 0;

            for (int i = 0; i < 4000; i++) {
                Q2.x = P.x + t * v2.x;
                Q2.y = P.y + t * v2.y;
                Q2.z = P.z + t * v2.z;

                float currentRadiusSquare = (Q2.x- C.x)*(Q2.x- C.x) + (Q2.y- C.y)*(Q2.y- C.y);
                if ( abs(currentRadiusSquare - gelTubeRadiusSquare) < 1) break;

                if (abs(currentRadiusSquare - gelTubeRadiusSquare) > abs(lastRadiusSquare - gelTubeRadiusSquare)) {
                    if (iterationsAfterLastChange <= 2) { //local minimum
                        t = t - increment;
                        increment = -increment/10;
                    } else {
                        increment = -increment;
                    }
                    
                    iterationsAfterLastChange = -1;
                }

                t = t + increment;
                lastRadiusSquare = currentRadiusSquare;
                iterationsAfterLastChange++;
            }

            // save the ray traveled distance
            traveledLength = t;

            eps=0.000001;

            // v2*t is the distance from P to the next refraction point
            v2.x = v2.x * t;
            v2.y = v2.y * t;
            v2.z = v2.z * t;
            
            ray = v2;
            newSource = P; //current point

            Q1.x = newSource.x;
            Q1.y = newSource.y;
            Q1.z = newSource.z;
            Q2.x = newSource.x + ray.x;
            Q2.y = newSource.y + ray.y;
            Q2.z = newSource.z + ray.z;

            axm=0;
            aym=0;
            azm=0;
            axM=1;
            ayM=1;
            azM=1;
            
            // am=max(max(axm,aym),azm); // 0
            // aM=min(min(axM,ayM),azM); // 1
            am=0; // 0
            aM=1; // 1
            
            // line intersects voxel space ->   am<aM
            if (am>=aM){
                // detector[idx]=0;
                return;
            }

            if( ray.x > 0){
                imin=(am==axm)? Q1.x + 1.0f             : ceilf (newSource.x+am*ray.x);
                imax=(aM==axM)? Q2.x + 1.0f             : floorf(newSource.x+aM*ray.x);
            }else{
                imax=(am==axm)? Q1.x             : floorf(newSource.x+am*ray.x);
                imin=(aM==axM)? Q2.x             : ceilf (newSource.x+aM*ray.x);
            }
            // for Y
            if( ray.y > 0){
                jmin=(am==aym)? Q1.y + 1.0f             : ceilf (newSource.y+am*ray.y);
                jmax=(aM==ayM)? Q2.y + 1.0f             : floorf(newSource.y+aM*ray.y);
            }else{
                jmax=(am==aym)? Q1.y             : floorf(newSource.y+am*ray.y);
                jmin=(aM==ayM)? Q2.y             : ceilf (newSource.y+aM*ray.y);
            }
            // for Z
            if( ray.z > 0){
                kmin=(am==azm)? Q1.z + 1.0f             : ceilf (newSource.z+am*ray.z);
                kmax=(aM==azM)? Q2.z + 1.0f             : floorf(newSource.z+aM*ray.z);
            }else{
                kmax=(am==azm)? Q1.z             : floorf(newSource.z+am*ray.z);
                kmin=(aM==azM)? Q2.z             : ceilf (newSource.z+aM*ray.z);
            }

            // get intersection point N1. eq(20-21) [(also eq 9-10)]
            ax=(ray.x>0)?  __fdividef(imin-newSource.x,ray.x) :  __fdividef(imax-newSource.x,ray.x);
            ay=(ray.y>0)?  __fdividef(jmin-newSource.y,ray.y) :  __fdividef(jmax-newSource.y,ray.y);
            az=(ray.z>0)?  __fdividef(kmin-newSource.z,ray.z) :  __fdividef(kmax-newSource.z,ray.z);
            
            // If its Infinite (i.e. ray is perpendicular to this axis), make sure its positive
            ax=(isinf(ax))? abs(ax) : ax;
            ay=(isinf(ay))? abs(ay) : ay;
            az=(isinf(az))? abs(az) : az;    
            
            // get index of first intersection. eq (26) and (19)
            aminc=min(min(ax,ay),az);
            i=floor(newSource.x+ (aminc+am)*0.5f*ray.x);
            j=floor(newSource.y+ (aminc+am)*0.5f*ray.y);
            k=floor(newSource.z+ (aminc+am)*0.5f*ray.z);
            // Initialize
            ac=am;
            //eq (28), unit anlges
            axu=__frcp_rd(abs(ray.x));
            ayu=__frcp_rd(abs(ray.y));
            azu=__frcp_rd(abs(ray.z));

            // eq(29), direction of update
            iu=(ray.x > 0)? 1.0f : -1.0f;
            ju=(ray.y > 0)? 1.0f : -1.0f;
            ku=(ray.z > 0)? 1.0f : -1.0f;

            unsigned long Np=(abs(imax-imin)+1)+(abs(jmax-jmin)+1)+(abs(kmax-kmin)+1); // Number of intersections

            // Go iterating over the line, intersection by intersection. If double point, no worries, 0 will be computed
            i+=0.5f*iu;
            j+=0.5f*ju;
            k+=0.5f*ku;
            ax+=axu;
            ay+=ayu;
            az+=azu;
            aminc=min(min(ax,ay),az);

        #pragma unroll
            for (unsigned long interactions = 0; interactions < Np; interactions++){
                if (ax==aminc){
                    sum+=(ax-ac)*tex3D<float>(tex, i, j, k);
                    // sum+=tex3D<float>(tex, i, j, k);
                    i=i+iu;
                    ac=ax;
                    ax+=axu;
                }else if(ay==aminc){
                    sum+=(ay-ac)*tex3D<float>(tex, i, j, k);
                    // sum+=tex3D<float>(tex, i, j, k);
                    j=j+ju;
                    ac=ay;
                    ay+=ayu;
                }else if(az==aminc){
                    sum+=(az-ac)*tex3D<float>(tex, i, j, k);
                    // sum+=tex3D<float>(tex, i, j, k);
                    k=k+ku;
                    ac=az;
                    az+=azu;
                }
            
                aminc=min(min(ax,ay),az);
            }

            float resSum = sum*traveledLength*geo.dVoxelX;

            detector[idx] = resSum;
}


int siddon_ray_projection(float* img, Geometry geo, float** result,float const * const angles,int nangles, const GpuIds& gpuids){
    // Prepare for MultiGPU
    int deviceCount = gpuids.GetLength();
    cudaCheckErrors("Device query fail");
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("Ax:Siddon_projection:GPUselect","There are no available device(s) that support CUDA\n");
    }
    //
    // CODE assumes
    // 1.-All available devices are usable by this code
    // 2.-All available devices are equal, they are the same machine (warning thrown)
    // Check the available devices, and if they are the same
    if (!gpuids.AreEqualDevices()) {
        mexWarnMsgIdAndTxt("Ax:Siddon_projection:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
    }
    int dev;
    
    // Check free memory
    size_t mem_GPU_global;
    checkFreeMemory(gpuids, &mem_GPU_global);

    size_t mem_image=                 (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY*(unsigned long long)geo.nVoxelZ*sizeof(float);
    size_t mem_proj=                  (unsigned long long)geo.nDetecU*(unsigned long long)geo.nDetecV*sizeof(float);
    
    // Does everything fit in the GPUs?
    const bool fits_in_memory = mem_image+2*PROJ_PER_BLOCK*mem_proj<mem_GPU_global;
    const char* boolStr = fits_in_memory ? "true" : "false";
    unsigned int splits=1;
    if (!fits_in_memory) {
        // Nope nope.
        // approx free memory we have. We already have left some extra 5% free for internal stuff
        // we need a second projection memory to combine multi-GPU stuff.
        size_t mem_free=mem_GPU_global-4*PROJ_PER_BLOCK*mem_proj;
        splits=mem_image/mem_free+1;// Ceil of the truncation
    }
    Geometry* geoArray = (Geometry*)malloc(splits*sizeof(Geometry));
    splitImage(splits,geo,geoArray,nangles);
    
    // Allocate axuiliary memory for projections on the GPU to accumulate partial results
    float ** dProjection_accum;
    size_t num_bytes_proj = PROJ_PER_BLOCK*geo.nDetecU*geo.nDetecV * sizeof(float);
    if (!fits_in_memory){
        dProjection_accum=(float**)malloc(2*deviceCount*sizeof(float*));
        for (dev = 0; dev < deviceCount; dev++) {
            cudaSetDevice(gpuids[dev]);
            for (int i = 0; i < 2; ++i){
                cudaMalloc((void**)&dProjection_accum[dev*2+i], num_bytes_proj);
                cudaMemset(dProjection_accum[dev*2+i],0,num_bytes_proj);
                cudaCheckErrors("cudaMallocauxiliarty projections fail");
            }
        }
    }
    
    // This is happening regarthless if the image fits on memory
    float** dProjection=(float**)malloc(2*deviceCount*sizeof(float*));
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        
        for (int i = 0; i < 2; ++i){
            cudaMalloc((void**)&dProjection[dev*2+i],   num_bytes_proj);
            cudaMemset(dProjection[dev*2+i]  ,0,num_bytes_proj);
            cudaCheckErrors("cudaMalloc projections fail");
        }
    }
    
    
    //Pagelock memory for synchronous copy.
    // Lets try to make the host memory pinned:
    // We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
    int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
    cudaDeviceGetAttribute(&isHostRegisterSupported,cudaDevAttrHostRegisterSupported,gpuids[0]);
#endif
    // empirical testing shows that when the image split is smaller than 1 (also implies the image is not very big), the time to
    // pin the memory is greater than the lost time in Synchronously launching the memcpys. This is only worth it when the image is too big.
#ifndef NO_PINNED_MEMORY
    if (isHostRegisterSupported & (splits>1 |deviceCount>1)){
        cudaHostRegister(img, (size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geo.nVoxelZ*(size_t)sizeof(float),cudaHostRegisterPortable);
    }
#endif
    cudaCheckErrors("Error pinning memory");

    
    
    // auxiliary variables
    Point3D source, deltaU, deltaV, uvOrigin;
    Point3D* projParamsArrayHost;
    cudaMallocHost((void**)&projParamsArrayHost, 6 * PROJ_PER_BLOCK * sizeof(Point3D));
    cudaCheckErrors("Error allocating auxiliary constant memory");
    
    // Create Streams for overlapping memcopy and compute
    int nStreams=deviceCount*2;
    cudaStream_t* stream=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));;
    
    
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        for (int i = 0; i < 2; ++i){
            cudaStreamCreate(&stream[i+dev*2]);
            
        }
    }
    cudaCheckErrors("Stream creation fail");

    int nangles_device=(nangles+deviceCount-1)/deviceCount;
    int nangles_last_device=(nangles-(deviceCount-1)*nangles_device);
    unsigned int noOfKernelCalls = (nangles_device+PROJ_PER_BLOCK-1)/PROJ_PER_BLOCK;  // We'll take care of bounds checking inside the loop if nalpha is not divisible by PROJ_PER_BLOCK
    unsigned int noOfKernelCallsLastDev = (nangles_last_device+PROJ_PER_BLOCK-1)/PROJ_PER_BLOCK; // we will use this in the memory management.
    int projection_this_block;
    cudaTextureObject_t *texImg = new cudaTextureObject_t[deviceCount];
    cudaArray **d_cuArrTex = new cudaArray*[deviceCount];
    
    for (unsigned int sp=0;sp<splits;sp++){
        
        // Create texture objects for all GPUs
        
        
        size_t linear_idx_start;
        //First one should always be  the same size as all the rest but the last
        linear_idx_start= (size_t)sp*(size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
        
        
        CreateTexture(gpuids,&img[linear_idx_start],geoArray[sp],d_cuArrTex,texImg,!sp);
        cudaCheckErrors("Texture object creation fail");
        
        
        // Prepare kernel lauch variables
        
        int divU,divV;
        divU=PIXEL_SIZE_BLOCK/8;
        divV=PIXEL_SIZE_BLOCK;
        dim3 blocks(ceil((2*geoArray[sp].gelTubeRadius/geoArray[sp].dVoxelY+divU-1)/divU),ceil((geoArray[sp].nDetecV+divV-1)/divV), 1);
        dim3 threadsPerBlock(divU,divV,PROJ_PER_BLOCK);
        
        unsigned int proj_global;
        // Now that we have prepared the image (piece of image) and parameters for kernels
        // we project for all angles.

        for (unsigned int i=0; i<noOfKernelCalls; i++) { // Iterating over all 720 projs / PROJ_PER_BLOCK
            for (dev=0;dev<deviceCount;dev++){
                cudaSetDevice(gpuids[dev]);
                
                for(unsigned int j=0; j<PROJ_PER_BLOCK; j++){
                    proj_global=(i*PROJ_PER_BLOCK+j)+dev*nangles_device;
                    if (proj_global>=nangles)
                        break;
                    if ((i*PROJ_PER_BLOCK+j)>=nangles_device)
                        break;
                    geoArray[sp].alpha=angles[proj_global*3];
                    geoArray[sp].theta=angles[proj_global*3+1];
                    geoArray[sp].psi  =angles[proj_global*3+2];
                    Point3D aux, aux2;
                    aux.x = geoArray[sp].DSD[proj_global];
                    aux.y = geoArray[sp].DSO[proj_global];
                    aux.z = geoArray[sp].EPS;
                    aux2.x = geoArray[sp].gelTubeRadius;

                    //precomute distances for faster execution
                    //Precompute per angle constant stuff for speed
                    computeDeltas_Siddon(geoArray[sp], proj_global, &uvOrigin, &deltaU, &deltaV, &source);
                    //Ray tracing!
                    projParamsArrayHost[6 * j] = uvOrigin;		// 6*j because we have 6 Point3D values per projection
                    projParamsArrayHost[6 * j + 1] = deltaU;
                    projParamsArrayHost[6 * j + 2] = deltaV;
                    projParamsArrayHost[6 * j + 3] = source;
                    projParamsArrayHost[6 * j + 4] = aux;
                    projParamsArrayHost[6 * j + 5] = aux2;
                }
                cudaMemcpyToSymbolAsync(projParamsArrayDev, projParamsArrayHost, sizeof(Point3D)*6*PROJ_PER_BLOCK,0,cudaMemcpyHostToDevice,stream[dev*2]);
                cudaStreamSynchronize(stream[dev*2]);
                cudaCheckErrors("kernel fail");
                kernelPixelDetector<<<blocks,threadsPerBlock,0,stream[dev*2]>>>(geoArray[sp],dProjection[(i%2)+dev*2],i,nangles_device,texImg[dev]);
            }

            // Now that the computation is happening, we need to either prepare the memory for
            // combining of the projections (splits>1) and start removing previous results.
            
            
            // If our image does not fit in memory then we need to make sure we accumulate previous results too.
            // This is done in 2 steps: 
            // 1)copy previous results back into GPU 
            // 2)accumulate with current results
            // The code to take them out is the same as when there are no splits needed
            if( !fits_in_memory&&sp>0)
            {
                // 1) grab previous results and put them in the auxiliary variable dProjection_accum
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on this set on this GPU
                    proj_global=i*PROJ_PER_BLOCK+dev*nangles_device;
                    if(proj_global>=nangles) 
                        break;

                    // Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
                    if(i+1==noOfKernelCalls) //is it the last block?
                        projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                                  nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
                    else
                        projection_this_block=PROJ_PER_BLOCK;

                    cudaMemcpyAsync(dProjection_accum[(i%2)+dev*2], result[proj_global], projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyHostToDevice,stream[dev*2+1]);
                }
                //  2) take the results from current compute call and add it to the code in execution.
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on this set on this GPU
                    proj_global=i*PROJ_PER_BLOCK+dev*nangles_device;
                    if(proj_global>=nangles) 
                        break;

                    // Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
                    if(i+1==noOfKernelCalls) //is it the last block?
                        projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                                  nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
                    else
                        projection_this_block=PROJ_PER_BLOCK;

                    cudaStreamSynchronize(stream[dev*2+1]); // wait until copy is finished
                    vecAddInPlace<<<(geo.nDetecU*geo.nDetecV*projection_this_block+MAXTREADS-1)/MAXTREADS,MAXTREADS,0,stream[dev*2]>>>(dProjection[(i%2)+dev*2],dProjection_accum[(i%2)+dev*2],(unsigned long)geo.nDetecU*geo.nDetecV*projection_this_block);
                }
            } // end accumulation case, where the image needs to be split 

            // Now, lets get out the projections from the previous execution of the kernels.
            if (i>0){
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on previous set on this GPU
                    proj_global=(i-1)*PROJ_PER_BLOCK+dev*nangles_device;
                    if (dev+1==deviceCount) {    //is it the last device?
                        // projections assigned to this device is >=nangles_device-(deviceCount-1) and < nangles_device
                        if (i-1 < noOfKernelCallsLastDev) {
                            // The previous set(block) was not empty.
                            projection_this_block=min(PROJ_PER_BLOCK, nangles-proj_global);
                        }
                        else {
                            // The previous set was empty.
                            // This happens if deviceCount > PROJ_PER_BLOCK+1.
                            // e.g. PROJ_PER_BLOCK = 9, deviceCount = 11, nangles = 199.
                            // e.g. PROJ_PER_BLOCK = 1, deviceCount =  3, nangles =   7.
                            break;
                        }
                    }
                    else {
                        projection_this_block=PROJ_PER_BLOCK;
                    }
                    cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(i%2))+dev*2],  projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*2+1]);
                }
            }
            // Make sure Computation on kernels has finished before we launch the next batch.
            for (dev = 0; dev < deviceCount; dev++){
                cudaSetDevice(gpuids[dev]);
                cudaStreamSynchronize(stream[dev*2]);
            }
        }
        
         // We still have the last set of projections to get out of GPUs
        for (dev = 0; dev < deviceCount; dev++)
        {
            cudaSetDevice(gpuids[dev]);
            //Global index of FIRST projection on this set on this GPU
            proj_global=(noOfKernelCalls-1)*PROJ_PER_BLOCK+dev*nangles_device;
            if(proj_global>=nangles) 
                break;
            // How many projections are left here?
            projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                      nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)

            cudaDeviceSynchronize(); //Not really necesary, but just in case, we los nothing. 
            cudaCheckErrors("Error at copying the last set of projections out (or in the previous copy)");
            cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(noOfKernelCalls%2))+dev*2], projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*2+1]);
        }
        // Make sure everyone has done their bussiness before the next image split:
        cudaDeviceSynchronize();

    } // End image split loop.
    
    cudaCheckErrors("Main loop  fail");
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaDestroyTextureObject(texImg[dev]);
            cudaFreeArray(d_cuArrTex[dev]);
    }
    delete[] texImg; texImg = 0;
    delete[] d_cuArrTex; d_cuArrTex = 0;
    // Freeing Stage
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaFree(dProjection[dev*2]);
        cudaFree(dProjection[dev*2+1]);
        
    }
    free(dProjection);
    
    if(!fits_in_memory){
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaFree(dProjection_accum[dev*2]);
            cudaFree(dProjection_accum[dev*2+1]);
            
        }
        free(dProjection_accum);
    }
    freeGeoArray(splits,geoArray);
    cudaFreeHost(projParamsArrayHost);
   
    
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]) ;
#ifndef NO_PINNED_MEMORY
    if (isHostRegisterSupported & (splits>1 |deviceCount>1)){
        cudaHostUnregister(img);
    }
    cudaCheckErrors("cudaFree  fail");
#endif
    cudaDeviceReset();
    return 0;
}




void CreateTexture(const GpuIds& gpuids,const float* imagedata,Geometry geo,cudaArray** d_cuArrTex, cudaTextureObject_t *texImage,bool alloc)
{
    //size_t size_image=geo.nVoxelX*geo.nVoxelY*geo.nVoxelZ;
    const cudaExtent extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
    const unsigned int num_devices = gpuids.GetLength();
    if(alloc){
        for (unsigned int dev = 0; dev < num_devices; dev++){
            cudaSetDevice(gpuids[dev]);
            
            //cudaArray Descriptor
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            //cuda Array
            cudaMalloc3DArray(&d_cuArrTex[dev], &channelDesc, extent);
        }
    }
    for (unsigned int dev = 0; dev < num_devices; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaMemcpy3DParms copyParams = {0};
        //Array creation
        copyParams.srcPtr   = make_cudaPitchedPtr((void *)imagedata, extent.width*sizeof(float), extent.width, extent.height);
        copyParams.dstArray = d_cuArrTex[dev];
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3DAsync(&copyParams);
    }
    for (unsigned int dev = 0; dev < num_devices; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaResourceDesc    texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array  = d_cuArrTex[dev];
        cudaTextureDesc     texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeBorder;
        texDescr.addressMode[1] = cudaAddressModeBorder;
        texDescr.addressMode[2] = cudaAddressModeBorder;
        texDescr.readMode = cudaReadModeElementType;
        cudaCreateTextureObject(&texImage[dev], &texRes, &texDescr, NULL);
        
    }
    for (unsigned int dev = 0; dev < num_devices; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaDeviceSynchronize();
    }
    cudaCheckErrors("Texture object creation fail");
}

/* This code generates the geometries needed to split the image properly in
 * cases where the entire image does not fit in the memory of the GPU
 **/
void splitImage(unsigned int splits,Geometry geo,Geometry* geoArray, unsigned int nangles){
    
    unsigned long splitsize=(geo.nVoxelZ+splits-1)/splits;// ceil if not divisible
    for(unsigned int sp=0;sp<splits;sp++){
        geoArray[sp]=geo;
        // All of them are splitsize, but the last one, possible
        geoArray[sp].nVoxelZ=((sp+1)*splitsize<geo.nVoxelZ)?  splitsize:  geo.nVoxelZ-splitsize*sp;
        geoArray[sp].sVoxelZ= geoArray[sp].nVoxelZ* geoArray[sp].dVoxelZ;
        
        // We need to redefine the offsets, as now each subimage is not aligned in the origin.
        geoArray[sp].offOrigZ=(float *)malloc(nangles*sizeof(float));
        for (unsigned int i=0;i<nangles;i++){
            geoArray[sp].offOrigZ[i]=geo.offOrigZ[i]-geo.sVoxelZ/2+sp*geoArray[0].sVoxelZ+geoArray[sp].sVoxelZ/2;
        }
        
    }
    
}

/* This code precomputes The location of the source and the Delta U and delta V (in the warped space)
 * to compute the locations of the x-rays. While it seems verbose and overly-optimized,
 * it does saves about 30% of each of the kernel calls. Thats something!
 **/
void computeDeltas_Siddon(Geometry geo,int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source){

    
    Point3D S;
    S.x=geo.DSO[i];
    S.y=0;
    S.z=0;
    
    //End point
    Point3D P,Pu0,Pv0;
    
    P.x  =-(geo.DSD[i]-geo.DSO[i]);   P.y  = geo.dDetecU*(0-((float)geo.nDetecU/2)+0.5);       P.z  = geo.dDetecV*(((float)geo.nDetecV/2)-0.5-0);
    Pu0.x=-(geo.DSD[i]-geo.DSO[i]);   Pu0.y= geo.dDetecU*(1-((float)geo.nDetecU/2)+0.5);       Pu0.z= geo.dDetecV*(((float)geo.nDetecV/2)-0.5-0);
    Pv0.x=-(geo.DSD[i]-geo.DSO[i]);   Pv0.y= geo.dDetecU*(0-((float)geo.nDetecU/2)+0.5);       Pv0.z= geo.dDetecV*(((float)geo.nDetecV/2)-0.5-1);
    // Geomtric trasnformations:
    // Now we have the Real world (OXYZ) coordinates of the UPPER LEFT corner and its two neighbours.
    // The obkjective is to get a position of the detector in a coordinate system where:
    // 1-units are voxel size (in each direction can be different)
    // 2-The image has the its first voxel at (0,0,0)
    // 3-The image never rotates
    
    // To do that, we need to compute the "deltas" the detector, or "by how much
    // (in new xyz) does the voxels change when and index is added". To do that
    // several geometric steps needs to be changed
    
    //1.Roll,pitch,jaw
    // The detector can have a small rotation.
    // according to
    //"A geometric calibration method for cone beam CT systems" Yang K1, Kwan AL, Miller DF, Boone JM. Med Phys. 2006 Jun;33(6):1695-706.
    // Only the Z rotation will have a big influence in the image quality when they are small.
    // Still all rotations are supported
    
    // To roll pitch jaw, the detector has to be in centered in OXYZ.
    P.x=0;Pu0.x=0;Pv0.x=0;
    
    // Roll pitch yaw
    rollPitchYaw(geo,i,&P);
    rollPitchYaw(geo,i,&Pu0);
    rollPitchYaw(geo,i,&Pv0);
    //Now ltes translate the points where they should be:
    P.x=P.x-(geo.DSD[i]-geo.DSO[i]);
    Pu0.x=Pu0.x-(geo.DSD[i]-geo.DSO[i]);
    Pv0.x=Pv0.x-(geo.DSD[i]-geo.DSO[i]);
    
    //1: Offset detector
    
    
    //S doesnt need to chagne
    
    
    //3: Rotate (around z)!
    Point3D Pfinal, Pfinalu0, Pfinalv0;
    Pfinal.x  =P.x;
    Pfinal.y  =P.y  +geo.offDetecU[i]; Pfinal.z  =P.z  +geo.offDetecV[i];
    Pfinalu0.x=Pu0.x;
    Pfinalu0.y=Pu0.y  +geo.offDetecU[i]; Pfinalu0.z  =Pu0.z  +geo.offDetecV[i];
    Pfinalv0.x=Pv0.x;
    Pfinalv0.y=Pv0.y  +geo.offDetecU[i]; Pfinalv0.z  =Pv0.z  +geo.offDetecV[i];
    
    eulerZYZ(geo,&Pfinal);
    eulerZYZ(geo,&Pfinalu0);
    eulerZYZ(geo,&Pfinalv0);
    eulerZYZ(geo,&S);
    
    //2: Offset image (instead of offseting image, -offset everything else)
    
    Pfinal.x  =Pfinal.x-geo.offOrigX[i];     Pfinal.y  =Pfinal.y-geo.offOrigY[i];     Pfinal.z  =Pfinal.z-geo.offOrigZ[i];
    Pfinalu0.x=Pfinalu0.x-geo.offOrigX[i];   Pfinalu0.y=Pfinalu0.y-geo.offOrigY[i];   Pfinalu0.z=Pfinalu0.z-geo.offOrigZ[i];
    Pfinalv0.x=Pfinalv0.x-geo.offOrigX[i];   Pfinalv0.y=Pfinalv0.y-geo.offOrigY[i];   Pfinalv0.z=Pfinalv0.z-geo.offOrigZ[i];
    S.x=S.x-geo.offOrigX[i];               S.y=S.y-geo.offOrigY[i];               S.z=S.z-geo.offOrigZ[i];
    
    // As we want the (0,0,0) to be in a corner of the image, we need to translate everything (after rotation);
    Pfinal.x  =Pfinal.x+geo.sVoxelX/2;      Pfinal.y  =Pfinal.y+geo.sVoxelY/2;          Pfinal.z  =Pfinal.z  +geo.sVoxelZ/2;
    Pfinalu0.x=Pfinalu0.x+geo.sVoxelX/2;    Pfinalu0.y=Pfinalu0.y+geo.sVoxelY/2;        Pfinalu0.z=Pfinalu0.z+geo.sVoxelZ/2;
    Pfinalv0.x=Pfinalv0.x+geo.sVoxelX/2;    Pfinalv0.y=Pfinalv0.y+geo.sVoxelY/2;        Pfinalv0.z=Pfinalv0.z+geo.sVoxelZ/2;
    S.x      =S.x+geo.sVoxelX/2;          S.y      =S.y+geo.sVoxelY/2;              S.z      =S.z      +geo.sVoxelZ/2;
    
    //4. Scale everything so dVoxel==1
    Pfinal.x  =Pfinal.x/geo.dVoxelX;      Pfinal.y  =Pfinal.y/geo.dVoxelY;        Pfinal.z  =Pfinal.z/geo.dVoxelZ;
    Pfinalu0.x=Pfinalu0.x/geo.dVoxelX;    Pfinalu0.y=Pfinalu0.y/geo.dVoxelY;      Pfinalu0.z=Pfinalu0.z/geo.dVoxelZ;
    Pfinalv0.x=Pfinalv0.x/geo.dVoxelX;    Pfinalv0.y=Pfinalv0.y/geo.dVoxelY;      Pfinalv0.z=Pfinalv0.z/geo.dVoxelZ;
    S.x      =S.x/geo.dVoxelX;          S.y      =S.y/geo.dVoxelY;            S.z      =S.z/geo.dVoxelZ;
    
    
    //mexPrintf("COR: %f \n",geo.COR[i]);
    //5. apply COR. Wherever everything was, now its offesetd by a bit
    float CORx, CORy;
    CORx=-geo.COR[i]*sin(geo.alpha)/geo.dVoxelX;
    CORy= geo.COR[i]*cos(geo.alpha)/geo.dVoxelY;
    Pfinal.x+=CORx;   Pfinal.y+=CORy;
    Pfinalu0.x+=CORx;   Pfinalu0.y+=CORy;
    Pfinalv0.x+=CORx;   Pfinalv0.y+=CORy;
    S.x+=CORx; S.y+=CORy;
    
    // return
    
    *uvorigin=Pfinal;
    
    deltaU->x=Pfinalu0.x-Pfinal.x;
    deltaU->y=Pfinalu0.y-Pfinal.y;
    deltaU->z=Pfinalu0.z-Pfinal.z;
    
    deltaV->x=Pfinalv0.x-Pfinal.x;
    deltaV->y=Pfinalv0.y-Pfinal.y;
    deltaV->z=Pfinalv0.z-Pfinal.z;
    
    *source=S;
}


#ifndef PROJECTION_HPP

float maxDistanceCubeXY(Geometry geo, float alpha,int i){
    ///////////
    // Compute initial "t" so we access safely as less as out of bounds as possible.
    //////////
    
    
    float maxCubX,maxCubY;
    // Forgetting Z, compute max distance: diagonal+offset
    maxCubX=(geo.sVoxelX/2+ abs(geo.offOrigX[i]))/geo.dVoxelX;
    maxCubY=(geo.sVoxelY/2+ abs(geo.offOrigY[i]))/geo.dVoxelY;
    
    return geo.DSO[i]/geo.dVoxelX-sqrt(maxCubX*maxCubX+maxCubY*maxCubY);
    
}
void rollPitchYaw(Geometry geo,int i, Point3D* point){
    Point3D auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=cos(geo.dRoll[i])*cos(geo.dPitch[i])*auxPoint.x
            +(cos(geo.dRoll[i])*sin(geo.dPitch[i])*sin(geo.dYaw[i]) - sin(geo.dRoll[i])*cos(geo.dYaw[i]))*auxPoint.y
            +(cos(geo.dRoll[i])*sin(geo.dPitch[i])*cos(geo.dYaw[i]) + sin(geo.dRoll[i])*sin(geo.dYaw[i]))*auxPoint.z;
    
    point->y=sin(geo.dRoll[i])*cos(geo.dPitch[i])*auxPoint.x
            +(sin(geo.dRoll[i])*sin(geo.dPitch[i])*sin(geo.dYaw[i]) + cos(geo.dRoll[i])*cos(geo.dYaw[i]))*auxPoint.y
            +(sin(geo.dRoll[i])*sin(geo.dPitch[i])*cos(geo.dYaw[i]) - cos(geo.dRoll[i])*sin(geo.dYaw[i]))*auxPoint.z;
    
    point->z=-sin(geo.dPitch[i])*auxPoint.x
            +cos(geo.dPitch[i])*sin(geo.dYaw[i])*auxPoint.y
            +cos(geo.dPitch[i])*cos(geo.dYaw[i])*auxPoint.z;
    
}
void eulerZYZ(Geometry geo, Point3D* point){
    Point3D auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=(+cos(geo.alpha)*cos(geo.theta)*cos(geo.psi)-sin(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-cos(geo.alpha)*cos(geo.theta)*sin(geo.psi)-sin(geo.alpha)*cos(geo.psi))*auxPoint.y+
            cos(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->y=(+sin(geo.alpha)*cos(geo.theta)*cos(geo.psi)+cos(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-sin(geo.alpha)*cos(geo.theta)*sin(geo.psi)+cos(geo.alpha)*cos(geo.psi))*auxPoint.y+
            sin(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->z=-sin(geo.theta)*cos(geo.psi)*auxPoint.x+
            sin(geo.theta)*sin(geo.psi)*auxPoint.y+
            cos(geo.theta)*auxPoint.z;
    
    
}
//______________________________________________________________________________
//
//      Function:       freeGeoArray
//
//      Description:    Frees the memory from the geometry array for multiGPU.
//______________________________________________________________________________
void freeGeoArray(unsigned int splits,Geometry* geoArray){
    for(unsigned int sp=0;sp<splits;sp++){
        free(geoArray[sp].offOrigZ);
    }
    free(geoArray);
}
//______________________________________________________________________________
//
//      Function:       checkFreeMemory
//
//      Description:    check available memory on devices
//______________________________________________________________________________
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global){
    size_t memfree;
    size_t memtotal;
    const int deviceCount = gpuids.GetLength();

    for (int dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaMemGetInfo(&memfree,&memtotal);
        if(dev==0) *mem_GPU_global=memfree;
        if(memfree<memtotal/2){
            mexErrMsgIdAndTxt("Ax:Siddon_projection:GPUmemory","One (or more) of your GPUs is being heavily used by another program (possibly graphics-based).\n Free the GPU to run TIGRE\n");
        }
        cudaCheckErrors("Check mem error");
        
        *mem_GPU_global=(memfree<*mem_GPU_global)?memfree:*mem_GPU_global;
    }
    *mem_GPU_global=(size_t)((double)*mem_GPU_global*0.95);
    
    //*mem_GPU_global= insert your known number here, in bytes.
}
#endif
