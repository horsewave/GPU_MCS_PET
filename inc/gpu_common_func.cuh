/*
 * gpu_common_func.h
 *
 *  Created on: Jul 20, 2018
 *      Author: bma
 */

#ifndef _GPU_COMMON_FUNC_H_
#define _GPU_COMMON_FUNC_H_


//CUDA RunTime API


#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include "structure_device.h"
//#include "gpupetkernel.cuh"

// Wrapping for GPU functions
void wrap_set_device(int);
void wrap_copy_materials_to_device(GPUPhantomMaterials, GPUPhantomMaterials &);
void wrap_copy_phantom_to_device(GPUPhantom, GPUPhantom &);
void wrap_copy_activities_to_device(GPUPhantomActivities, GPUPhantomActivities &);
void wrap_copy_scanner_to_device(GPUScanner, GPUScanner &);

void wrap_init_particle_stack(int, GPUParticleStack &, GPUParticleStack &,
                                   GPUParticleStack &, GPUParticleStack &);
int *wrap_init_counter(int);
void wrap_init_PRNG(int, int, int, GPUParticleStack &, GPUParticleStack &,
                                   GPUParticleStack &, GPUParticleStack &);

void wrap_voxelized_source_b2b(GPUParticleStack &, GPUParticleStack &, GPUPhantomActivities &,
                               float3, int, int);

void wrap_navigation_regular(GPUParticleStack &, GPUParticleStack &,
                             GPUPhantom &, GPUPhantomMaterials &, float,
                             int *, int, int);

void wrap_detection(GPUParticleStack &, GPUParticleStack &,
                             GPUScanner &, GPUPhantom &,
                             int, int);

void wrap_get_stack_gamma1(GPUParticleStack &, GPUParticleStack &);
void wrap_get_stack_gamma2(GPUParticleStack &, GPUParticleStack &);
int wrap_get_nb_of_simulated(int *);
void wrap_free_counters(int*, int*);
void wrap_free(GPUPhantomMaterials &, GPUPhantomActivities &,
               GPUPhantom &, GPUScanner &, GPUParticleStack &, GPUParticleStack &,
               GPUParticleStack &, GPUParticleStack &, int *, int *);

// Utils
void _stack_host_malloc(GPUParticleStack &, int);
void _stack_device_malloc(GPUParticleStack &, int);
void _stack_host_free(GPUParticleStack &);
void _stack_device_free(GPUParticleStack &);
void _phantom_device_free(GPUPhantom &);
void _scanner_device_free(GPUScanner &);
void _activities_device_free(GPUPhantomActivities &);
void _materials_device_free(GPUPhantomMaterials &);



#endif /* GPU_COMMON_FUNC_H_ */
