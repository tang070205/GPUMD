/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef USE_HIP // HIP for AMD card
#include <hipfft/hipfft.h>
#include <hip/hip_runtime.h>

// memory manipulation
#define gpuMalloc hipMalloc
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuGetSymbolAddress hipGetSymbolAddress
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset hipMemset

// error handling
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError

// device manipulation
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuDeviceSynchronize hipDeviceSynchronize

// stream
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy

// random numbers
#define gpurandState hiprandState
#define gpurand_normal_double hiprand_normal_double
#define gpurand_normal hiprand_normal
#define gpurand_init hiprand_init

// blas
#define gpublasHandle_t hipblasHandle_t
#define gpublasSgemv hipblasSgemv
#define gpublasSgemm hipblasSgemm
#define gpublasSdgmm hipblasSdgmm
#define gpublasDgemvBatched hipblasDgemvBatched
#define gpublasDestroy hipblasDestroy
#define gpublasCreate hipblasCreate
#define GPUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define GPUBLAS_OP_N HIPBLAS_OP_N
#define GPUBLAS_OP_T HIPBLAS_OP_T

// lapack
#define gpuDoubleComplex hipDoubleComplex
#define gpusolverDnHandle_t hipsolverDnHandle_t
#define gpusolverDnCreate hipsolverDnCreate
#define gpusolverDnDestroy hipsolverDnDestroy
#define gpusolverEigMode_t hipsolverEigMode_t
#define gpusolverFillMode_t hipsolverFillMode_t
#define GPUSOLVER_EIG_MODE_NOVECTOR HIPSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER HIPSOLVER_FILL_MODE_LOWER
#define gpusolverSyevjInfo_t hipsolverSyevjInfo_t
#define gpusolverDnCreateSyevjInfo hipsolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo hipsolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize hipsolverDnZheevj_bufferSize
#define gpusolverDnZheevj hipsolverDnZheevj
#define gpusolverDnZheevd_bufferSize hipsolverDnZheevd_bufferSize
#define gpusolverDnZheevd hipsolverDnZheevd
#define gpusolverDnDsyevj_bufferSize hipsolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj hipsolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize hipsolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched hipsolverDnZheevjBatched

// FFT
#define gpufftHandle hipfftHandle
#define gpufftComplex hipfftComplex
#define gpufftExecC2C hipfftExecC2C
#define gpufftPlan3d hipfftPlan3d 
#define gpufftPlanMany hipfftPlanMany
#define gpufftDestroy hipfftDestroy
#define GPUFFT_SUCCESS HIPFFT_SUCCESS
#define GPUFFT_C2C HIPFFT_C2C
#define GPUFFT_FORWARD HIPFFT_FORWARD
#define GPUFFT_INVERSE HIPFFT_BACKWARD

#elif defined(USE_MUSA) // MUSA for Moore Threads 

#include <musa_runtime.h>

// memory manipulation
#define gpuMalloc   musaMalloc
#define gpuMallocManaged musaMallocManaged
#define gpuFree musaFree
#define gpuMemcpy musaMemcpy
#define gpuMemcpyFromSymbol musaMemcpyFromSymbol
#define gpuMemcpyToSymbol musaMemcpyToSymbol
#define gpuGetSymbolAddress musaGetSymbolAddress
#define gpuMemcpyHostToDevice musaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
#define gpuMemcpyHostToHost musaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define gpuMemset musaMemset

// error handling
#define gpuError_t musaError_t
#define gpuSuccess musaSuccess
#define gpuGetErrorString musaGetErrorString
#define gpuGetLastError musaGetLastError

// device manipulation
#define gpuSetDevice musaSetDevice
#define gpuGetDeviceCount musaGetDeviceCount
#define gpuDeviceProp musaDeviceProp
#define gpuGetDeviceProperties musaGetDeviceProperties
#define gpuDeviceCanAccessPeer musaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define gpuDeviceSynchronize musaDeviceSynchronize

// stream
#define gpuStream_t musaStream_t
#define gpuStreamCreate musaStreamCreate
#define gpuStreamDestroy musaStreamDestroy

// random numbers
#define gpurandState murandState
#define gpurand_normal_double murand_normal_double
#define gpurand_normal murand_normal
#define gpurand_init murand_init

// blas
#define gpublasHandle_t mublasHandle_t
#define gpublasSgemv mublasSgemv
#define gpublasSgemm mublasSgemm
#define gpublasSdgmm mublasSdgmm
#define gpublasDgemv mublasDgemv
#define gpublasDgemvBatched mublasDgemvBatched
#define gpublasDestroy mublasDestroy
#define gpublasCreate mublasCreate
#define GPUBLAS_SIDE_LEFT MUBLAS_SIDE_LEFT
#define GPUBLAS_OP_N MUBLAS_OP_N
#define GPUBLAS_OP_T MUBLAS_OP_T

// lapack
#define gpuDoubleComplex muDoubleComplex
#define gpusolverDnHandle_t musolverDnHandle_t
#define gpusolverDnCreate musolverDnCreate
#define gpusolverDnDestroy musolverDnDestroy
#define gpusolverEigMode_t musolverEigMode_t
#define gpusolverFillMode_t mublasFillMode_t
#define GPUSOLVER_EIG_MODE_NOVECTOR MUSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR MUSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER MUBLAS_FILL_MODE_LOWER
// #define gpusolverSyevjInfo_t syevjInfo_t  
// #define gpusolverDnCreateSyevjInfo cusolverDnCreateSyevjInfo
// #define gpusolverDnDestroySyevjInfo cusolverDnDestroySyevjInfo
// #define gpusolverDnZheevj_bufferSize cusolverDnZheevj_bufferSize
#define gpusolverDnZheevj musolverDnZheevj
#define gpusolverDnZheevd_bufferSize musolverDnZheevd_bufferSize
#define gpusolverDnZheevd musolverDnZheevd
// #define gpusolverDnDsyevj_bufferSize cusolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj musolverDnDsyevj
// #define gpusolverDnZheevjBatched_bufferSize cusolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched musolverDnZheevjBatched

// FFT
#define gpufftHandle mufftHandle
#define gpufftComplex mufftComplex
#define gpufftExecC2C mufftExecC2C
#define gpufftPlan3d mufftPlan3d 
#define gpufftPlanMany mufftPlanMany
#define gpufftDestroy mufftDestroy
#define GPUFFT_SUCCESS MUFFT_SUCCESS
#define GPUFFT_C2C MUFFT_C2C
#define GPUFFT_FORWARD MUFFT_FORWARD
#define GPUFFT_INVERSE MUFFT_INVERSE

#elif defined(USE_MXMACA) // MXMACA for MetaX (沐曦)

#include <mc_runtime_api.h>

// memory manipulation
#define gpuMalloc   mcMalloc
#define gpuMallocManaged mcMallocManaged
#define gpuFree mcFree
#define gpuMemcpy mcMemcpy
#define gpuMemcpyFromSymbol mcMemcpyFromSymbol
#define gpuMemcpyToSymbol mcMemcpyToSymbol
#define gpuGetSymbolAddress mcGetSymbolAddress
#define gpuMemcpyHostToDevice mcMemcpyHostToDevice
#define gpuMemcpyDeviceToHost mcMemcpyDeviceToHost
#define gpuMemcpyHostToHost mcMemcpyHostToHost
#define gpuMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#define gpuMemset mcMemset

// error handling
#define gpuError_t mcError_t
#define gpuSuccess mcSuccess
#define gpuGetErrorString mcGetErrorString
#define gpuGetLastError mcGetLastError

// device manipulation
#define gpuSetDevice mcSetDevice
#define gpuGetDeviceCount mcGetDeviceCount
#define gpuDeviceProp mcDeviceProp_t
#define gpuGetDeviceProperties mcGetDeviceProperties
#define gpuDeviceCanAccessPeer mcDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess mcDeviceEnablePeerAccess
#define gpuDeviceSynchronize mcDeviceSynchronize

// stream
#define gpuStream_t mcStream_t
#define gpuStreamCreate mcStreamCreate
#define gpuStreamDestroy mcStreamDestroy

// random numbers
#define gpurandState mcrandState
#define gpurand_normal_double mcrand_normal_double
#define gpurand_normal mcrand_normal
#define gpurand_init mcrand_init

// blas
#define gpublasHandle_t mcblasHandle_t
#define gpublasSgemv mcblasSgemv
#define gpublasSgemm mcblasSgemm
#define gpublasSdgmm mcblasSdgmm
#define gpublasDgemv mcblasDgemv
#define gpublasDgemvBatched mcblasDgemvBatched
#define gpublasDestroy mcblasDestroy
#define gpublasCreate mcblasCreate
#define GPUBLAS_SIDE_LEFT MCBLAS_SIDE_LEFT
#define GPUBLAS_OP_N MCBLAS_OP_N
#define GPUBLAS_OP_T MCBLAS_OP_T

// lapack
#define gpuDoubleComplex mcDoubleComplex
#define gpusolverDnHandle_t mcsolverDnHandle_t
#define gpusolverDnCreate mcsolverDnCreate
#define gpusolverDnDestroy mcsolverDnDestroy
#define gpusolverEigMode_t mcsolverEigMode_t
#define gpusolverFillMode_t mcblasFillMode_t
#define GPUSOLVER_EIG_MODE_NOVECTOR MCSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR MCSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER MCBLAS_FILL_MODE_LOWER
#define gpusolverSyevjInfo_t syevjInfo_t  
#define gpusolverDnCreateSyevjInfo mcsolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo mcsolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize mcsolverDnZheevj_bufferSize
#define gpusolverDnZheevj mcsolverDnZheevj
#define gpusolverDnZheevd_bufferSize mcsolverDnZheevd_bufferSize
#define gpusolverDnZheevd mcsolverDnZheevd
#define gpusolverDnDsyevj_bufferSize ccsolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj mcsolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize mcsolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched mcsolverDnZheevjBatched

// FFT
#define gpufftHandle mcfftHandle
#define gpufftComplex mcfftComplex
#define gpufftExecC2C mcfftExecC2C
#define gpufftPlan3d mcfftPlan3d 
#define gpufftPlanMany mcfftPlanMany
#define gpufftDestroy mcfftDestroy
#define GPUFFT_SUCCESS MCFFT_SUCCESS
#define GPUFFT_C2C MCFFT_C2C
#define GPUFFT_FORWARD MCFFT_FORWARD
#define GPUFFT_INVERSE MCFFT_INVERSE

#else // CUDA for Nvidia card

// memory manipulation
#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuGetSymbolAddress cudaGetSymbolAddress
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset

// error handling
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError

// device manipulation
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuDeviceSynchronize cudaDeviceSynchronize

// stream
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy

// random numbers
#define gpurandState curandState
#define gpurand_normal_double curand_normal_double
#define gpurand_normal curand_normal
#define gpurand_init curand_init

// blas
#define gpublasHandle_t cublasHandle_t
#define gpublasSgemv cublasSgemv
#define gpublasSgemm cublasSgemm
#define gpublasSdgmm cublasSdgmm
#define gpublasDgemv cublasDgemv
#if (CUDA_VERSION >= 12000)
#define gpublasDgemvBatched cublasDgemvBatched
#endif
#define gpublasDestroy cublasDestroy
#define gpublasCreate cublasCreate
#define GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

// lapack
#define gpuDoubleComplex cuDoubleComplex
#define gpusolverDnHandle_t cusolverDnHandle_t
#define gpusolverDnCreate cusolverDnCreate
#define gpusolverDnDestroy cusolverDnDestroy
#define gpusolverEigMode_t cusolverEigMode_t
#define gpusolverFillMode_t cublasFillMode_t // why cublas?
#define GPUSOLVER_EIG_MODE_NOVECTOR CUSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER // why cublas?
#define gpusolverSyevjInfo_t syevjInfo_t                 // why not cusolverSyevjInfo_t?
#define gpusolverDnCreateSyevjInfo cusolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo cusolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize cusolverDnZheevj_bufferSize
#define gpusolverDnZheevj cusolverDnZheevj
#define gpusolverDnZheevd_bufferSize cusolverDnZheevd_bufferSize
#define gpusolverDnZheevd cusolverDnZheevd
#define gpusolverDnDsyevj_bufferSize cusolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj cusolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize cusolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched cusolverDnZheevjBatched

// FFT
#define gpufftHandle cufftHandle
#define gpufftComplex cufftComplex
#define gpufftExecC2C cufftExecC2C
#define gpufftPlan3d cufftPlan3d 
#define gpufftPlanMany cufftPlanMany
#define gpufftDestroy cufftDestroy
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

#endif
