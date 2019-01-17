// This file is part of Heracles
// 
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// Heracles Copyright (C) 2013 Julien Bert 

/*******************************
 *Name: gpupetkernel.c
 *Function: all the kernel functions
 *
 *Editor: Bo Ma
 *Time: 2016.10.20
 *Version:1.0
 *
 *Notices: this header file can be only included in the source files who call it(gpu_common_func.cu). It can not be included '
 *in the header file(gpu_common_func.cuh).Otherwise there will be errors which indicate "multiple definitions of the variables."
 *The reason is not clear by now.
 *
 * ****************************/


#ifndef GPUPETKERNEL_CUH
#define GPUPETKERNEL_CUH

///// Utils ///////////////////////////////////////////////////////////////

#include <curand.h>
#include <curand_kernel.h>
//CUDA RunTime API
#include <cuda_runtime.h>


#include "structure_device.h"



// Inverse vector
__device__ float3 vec3_inverse(float3 u);

// rotateUz, function from CLHEP
__device__ float3 rotateUz(float3 vector, float3 newUz);

// Return the next voxel boundary distance, it is used by the standard navigator

__device__ float get_boundary_voxel_by_raycasting(int4 vox, float3 p, float3 d, float3 res) ;

// Ray/AABB - Smits algorithm
__device__ float mc_hit_AABB(float px, float py, float pz,     // Ray definition (pos, dir) 
                  float dx, float dy, float dz,
                  float xmin, float xmax,           // AABB definition
                  float ymin, float ymax, 
                  float zmin, float zmax) ;

// Ray/OBB
__device__ float mc_hit_OBB(float px, float py, float pz, // Ray definition (pos, dir) 
                 float dx, float dy, float dz,
                 float xmin, float xmax,       // Slabs are defined in OBB's space
                 float ymin, float ymax, 
                 float zmin, float zmax,
                 float cx, float cy, float cz, // OBB center
                 float ux, float uy, float uz, // OBB orthognal space (u, v, w)
                 float vx, float vy, float vz,
                 float wx, float wy, float wz) ;

// Return the distance to the block boundary

__device__ float get_boundary_block_by_raycasting(float3 p, float3 d, float3 halfsize) ;

__device__ float Move_To_Next_Block(float3& position, float3& direction, int& currentid, GPUScanner scanner, float d_min, int id);

// Linear search
__device__ int linear_search(float *val, float key) ;
// Relative search
__device__ int relative_search(float *val, float key, int n) ;

// Binary search
__device__ int binary_search(float *val, float key, int n) ;


///// PRNG Brent xor256 //////////////////////////////////////////////////

// Init rand seed for GPU
__global__ void kernel_random_seed_init(GPUParticleStack stackpart,curandState *state);


// Brent PRNG integer version
__device__ unsigned long weyl;
__device__ unsigned long brent_int(unsigned int index, unsigned long *device_x_brent, unsigned long seed);


// Brent PRNG real version
__device__ double Brent_real(int index, unsigned long *device_x_brent, unsigned long seed);


// Init Brent seed
__global__ void kernel_brent_init(GPUParticleStack stackpart);
///// Physics ///////////////////////////////////////////////////////////////

///// PhotoElectric /////

// PhotoElectric Cross Section Per Atom (Standard)
__device__ float PhotoElec_CSPA(float E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__device__ float PhotoElec_CS(GPUPhantomMaterials materials, 
                              unsigned short int mat, float E);
// Photoelectric effect without photoelectron
__device__ void PhotoElec_SampleSecondaries(GPUParticleStack photons, 
                                            unsigned int id, int *ct_d_sim);

///// Compton /////

// Compton Cross Section Per Atom (Standard - Klein-Nishina)
__device__ float Compton_CSPA(float E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__device__ float Compton_CS(GPUPhantomMaterials materials, 
                                     unsigned short int mat, float E);

// Compton Scatter without secondary electron (Standard - Klein-Nishina)
__device__ void Compton_SampleSecondaries(GPUParticleStack photons, 
                                           unsigned int id, int *d_ct_sim);

// Compton Scatter without secondary electron (Standard - Klein-Nishina)
__device__ void Compton_SampleSecondaries_Detector(GPUParticleStack photons, 
                                           unsigned int id, float3& direction, 
                                           float &energy, bool &act) ;

///// Source ///////////////////////////////////////////////////////////////

// Voxelized source
__global__ void kernel_voxelized_source_b2b(GPUParticleStack d_g1, 
                                            GPUParticleStack d_g2,
                                            GPUPhantomActivities d_act,
                                            float3 phantom_size_in_mm) ;
    

///// Navigation ///////////////////////////////////////////////////////////////

// Photons - regular tracking

__global__ void kernel_navigation_regular(GPUParticleStack photons,
                                          GPUPhantom phantom,
                                          GPUPhantomMaterials materials,
										  float radius,
                                          int* d_ct_sim) ;
    


__global__ void kernel_detection(GPUParticleStack photons,
                                          GPUScanner scanner, GPUPhantom phantom);

    
__global__ void kernel_detection_physics(GPUParticleStack photons,
                                          GPUScanner scanner, GPUPhantom phantom);
    
#endif

