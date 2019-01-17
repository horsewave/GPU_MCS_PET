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

//#ifndef GPUPETLIB_CU
//#define GPUPETLIB_CU


#include <cuda.h>
#include <curand.h>

#include "../inc/gpu_common_func.cuh"
#include "../inc/gpupetkernel.cuh"
#include "../inc/utilslib.h"
#include "../inc/gpuErrchk.cuh"





//// Utils /////////////////////////////////////////////////////////////////////////

// Stack host allocation
void _stack_host_malloc(GPUParticleStack &phasespace, int stack_size) {
	phasespace.size = stack_size;
	unsigned int mem_phasespace_float = stack_size * sizeof(float);
	unsigned int mem_phasespace_uint = stack_size * sizeof(unsigned int);
   	unsigned int mem_phasespace_sint = stack_size * sizeof(short int);
	unsigned int mem_phasespace_char = stack_size * sizeof(char);
	
	phasespace.E = (float*)malloc(mem_phasespace_float);
	phasespace.dx = (float*)malloc(mem_phasespace_float);
	phasespace.dy = (float*)malloc(mem_phasespace_float);
	phasespace.dz = (float*)malloc(mem_phasespace_float);
	phasespace.px = (float*)malloc(mem_phasespace_float);
	phasespace.py = (float*)malloc(mem_phasespace_float);
	phasespace.pz = (float*)malloc(mem_phasespace_float);
	phasespace.tof = (float*)malloc(mem_phasespace_float);
	phasespace.seed = (unsigned int*)malloc(mem_phasespace_uint);
	phasespace.endsimu = (unsigned char*)malloc(mem_phasespace_char);
	phasespace.active = (unsigned char*)malloc(mem_phasespace_char);    
   	phasespace.crystalID = (short int*)malloc(mem_phasespace_sint);
    phasespace.Edeposit = (float*)malloc(mem_phasespace_float);
   	phasespace.nCompton = (short int*)malloc(mem_phasespace_sint);
   	phasespace.nCompton_crystal = (short int*)malloc(mem_phasespace_sint);
   	phasespace.nPhotoelectric_crystal = (short int*)malloc(mem_phasespace_sint);

}

// For PRNG Brent
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64)
#define r      (4*UINT64 + 8*UINT32)
// Stack device allocation
void _stack_device_malloc(GPUParticleStack &stackpart, int stack_size) {
	stackpart.size = stack_size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
   	unsigned int mem_stackpart_sint = stack_size * sizeof(short int);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_brent;
	if (r == 4) {mem_brent = stack_size * 6 * sizeof(unsigned long);}
	else {mem_brent = stack_size * 10 * sizeof(unsigned long);}

	cudaMalloc((void**) &stackpart.E, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dx, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dy, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dz, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.px, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.py, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.pz, mem_stackpart_float);	
	cudaMalloc((void**) &stackpart.tof, mem_stackpart_float);	
	cudaMalloc((void**) &stackpart.seed, mem_stackpart_uint);
	cudaMalloc((void**) &stackpart.table_x_brent, mem_brent);
	cudaMalloc((void**) &stackpart.endsimu, mem_stackpart_char);
   	cudaMalloc((void**) &stackpart.active, mem_stackpart_char);
   	cudaMalloc((void**) &stackpart.crystalID, mem_stackpart_sint);
	cudaMalloc((void**) &stackpart.Edeposit, mem_stackpart_float);
   	cudaMalloc((void**) &stackpart.nCompton, mem_stackpart_sint);
   	cudaMalloc((void**) &stackpart.nCompton_crystal, mem_stackpart_sint);
   	cudaMalloc((void**) &stackpart.nPhotoelectric_crystal, mem_stackpart_sint);
    
}
#undef UINT64
#undef UINT32
#undef r

// Free stack host memory
void _stack_host_free(GPUParticleStack &stack) {
	free(stack.E); 
	free(stack.dx); 
	free(stack.dy);
	free(stack.dz);
	free(stack.px);
	free(stack.py); 
	free(stack.pz); 
	free(stack.tof);
	free(stack.seed);
	free(stack.endsimu); 
   	free(stack.active); 
    free(stack.crystalID);
	free(stack.Edeposit);     
    free(stack.nCompton);
    free(stack.nCompton_crystal);
    free(stack.nPhotoelectric_crystal);
}

// Free stack device memory
void _stack_device_free(GPUParticleStack &stack) {
	cudaFree(stack.E);
	cudaFree(stack.dx);
	cudaFree(stack.dy);
	cudaFree(stack.dz);
	cudaFree(stack.px);
	cudaFree(stack.py);
	cudaFree(stack.pz);	
	cudaFree(stack.tof);	
	cudaFree(stack.seed);
	cudaFree(stack.table_x_brent);
	cudaFree(stack.endsimu);
   	cudaFree(stack.active);
    cudaFree(stack.crystalID);
	cudaFree(stack.Edeposit);    
    cudaFree(stack.nCompton);
    cudaFree(stack.nCompton_crystal);
    cudaFree(stack.nPhotoelectric_crystal);

}
    
// Free materials device memory
void _materials_device_free(GPUPhantomMaterials &mat) {
    cudaFree(mat.nb_elements);
    cudaFree(mat.index);
    cudaFree(mat.mixture);
    cudaFree(mat.atom_num_dens);
}
 
// Free phantom device memory
void _phantom_device_free(GPUPhantom &phan) {
    cudaFree(phan.data);    
}

// Free activities device memory
void _activities_device_free(GPUPhantomActivities &act) {
    cudaFree(act.act_index);
    cudaFree(act.act_cdf);
}

// Free scanner device memory
void _scanner_device_free(GPUScanner &scanner) {
    cudaFree(scanner.pos);
    cudaFree(scanner.v0);
    cudaFree(scanner.v1);
    cudaFree(scanner.v2);
}

// Set a GPU device
void wrap_set_device(int id) {cudaSetDevice(id);}


// Allocate and copy the materials structure to the GPU
void wrap_copy_materials_to_device(GPUPhantomMaterials h_mat, GPUPhantomMaterials &d_mat) {
    
    // Memory allocation
    unsigned int nb_mat = h_mat.nb_materials;
    unsigned int nb_elm = h_mat.nb_elements_total;

    unsigned int mem_mat_usi = nb_mat * sizeof(unsigned short int);
    unsigned int mem_elm_usi = nb_elm * sizeof(unsigned short int);
    unsigned int mem_elm_float = nb_elm * sizeof(float);
    
    cudaMalloc((void**) &d_mat.nb_elements, mem_mat_usi);
    cudaMalloc((void**) &d_mat.index, mem_mat_usi);
    cudaMalloc((void**) &d_mat.mixture, mem_elm_usi);
    cudaMalloc((void**) &d_mat.atom_num_dens, mem_elm_float);
    
    // Copy structure
    d_mat.nb_materials = nb_mat;
    d_mat.nb_elements_total = nb_elm;
    
    cudaMemcpy(d_mat.nb_elements,   h_mat.nb_elements, mem_mat_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat.index,         h_mat.index, mem_mat_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat.mixture,       h_mat.mixture, mem_elm_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat.atom_num_dens, h_mat.atom_num_dens, mem_elm_float, cudaMemcpyHostToDevice);

}

// Allocate and copy the phantom to the GPU device
void wrap_copy_phantom_to_device(GPUPhantom h_phan, GPUPhantom &d_phan) {
    
    // Allocation
	cudaMalloc((void**) &d_phan.data, h_phan.mem_data);

    // Copy
    d_phan.size_in_vox = h_phan.size_in_vox;
    d_phan.voxel_size = h_phan.voxel_size;
    d_phan.size_in_mm = h_phan.size_in_mm;
    d_phan.nb_voxel_slice = h_phan.nb_voxel_slice;
    d_phan.nb_voxel_volume = h_phan.nb_voxel_volume;
    d_phan.mem_data = h_phan.mem_data;
	
    cudaMemcpy(d_phan.data, h_phan.data, d_phan.mem_data, cudaMemcpyHostToDevice);
}

// Allocate and copy activities to the GPU device
void wrap_copy_activities_to_device(GPUPhantomActivities h_act, GPUPhantomActivities &d_act) {

    unsigned int mem_usi = h_act.nb_activities*sizeof(unsigned int);
    unsigned int mem_float = h_act.nb_activities*sizeof(float);

    // Allocation
    cudaMalloc((void**) &d_act.act_index, mem_usi);
    cudaMalloc((void**) &d_act.act_cdf, mem_float);

    // Copy
    d_act.nb_activities = h_act.nb_activities;
    d_act.tot_activity = h_act.tot_activity;
 	d_act.size_in_vox = h_act.size_in_vox;
    d_act.voxel_size = h_act.voxel_size;
    d_act.size_in_mm = h_act.size_in_mm;	
    cudaMemcpy(d_act.act_index, h_act.act_index, mem_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_act.act_cdf, h_act.act_cdf, mem_float, cudaMemcpyHostToDevice);

}

// Allocate and copy scanner to the GPU device
void wrap_copy_scanner_to_device(GPUScanner h_scan, GPUScanner &d_scanner) {

    unsigned int mem_float_pos= h_scan.nblock*h_scan.ncass*sizeof(float3);
    unsigned int mem_float_v= h_scan.ncass*sizeof(float3);
    
    // Allocation
    cudaMalloc((void**) &d_scanner.pos, mem_float_pos);
    cudaMalloc((void**) &d_scanner.v0, mem_float_v);
    cudaMalloc((void**) &d_scanner.v1, mem_float_v);
    cudaMalloc((void**) &d_scanner.v2, mem_float_v);

    // Copy
    d_scanner.cyl_radius=h_scan.cyl_radius;
    d_scanner.cyl_halfheight=h_scan.cyl_halfheight;
    d_scanner.block_pitch=h_scan.block_pitch;
    d_scanner.cta_pitch=h_scan.cta_pitch;
    d_scanner.cax_pitch=h_scan.cax_pitch;
    d_scanner.ncass=h_scan.ncass;
    d_scanner.nblock=h_scan.nblock;
    d_scanner.ncry_ta=h_scan.ncry_ta;
    d_scanner.ncry_ax=h_scan.ncry_ax;
    d_scanner.blocksize=h_scan.blocksize;
    d_scanner.halfsize.x=h_scan.halfsize.x;
    d_scanner.halfsize.y=h_scan.halfsize.y;
    d_scanner.halfsize.z=h_scan.halfsize.z;
    cudaMemcpy(d_scanner.pos, h_scan.pos, mem_float_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scanner.v0, h_scan.v0, mem_float_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scanner.v1, h_scan.v1, mem_float_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scanner.v2, h_scan.v2, mem_float_v, cudaMemcpyHostToDevice);
    wrap_copy_materials_to_device(h_scan.mat, d_scanner.mat);


    }

// Allocate particle stack
void wrap_init_particle_stack(int size, GPUParticleStack &h_g1, GPUParticleStack &h_g2,
                                        GPUParticleStack &d_g1, GPUParticleStack &d_g2) {
    // init stack on the host 
    _stack_host_malloc(h_g1, size);
    _stack_host_malloc(h_g2, size);

    // init stack on the device
    _stack_device_malloc(d_g1, size);
    _stack_device_malloc(d_g2, size);

    // set the endsimu flag to say that the stack is empty
    int i=0; while (i<h_g1.size) {
        h_g1.endsimu[i] = 1;
        h_g2.endsimu[i] = 1;
        ++i;
    }
    cudaMemcpy(d_g1.seed, h_g1.seed, sizeof(char)*h_g1.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g2.seed, h_g2.seed, sizeof(char)*h_g2.size, cudaMemcpyHostToDevice);
}

// Init counter
int *wrap_init_counter(int h_val) {
    int *d_val;
    cudaMalloc((void**) &d_val, sizeof(int));
    cudaMemcpy(d_val, &h_val, sizeof(int), cudaMemcpyHostToDevice);

    return d_val;
}

// Get back the number of simulated particle
int wrap_get_nb_of_simulated(int *d_ct_sim) {
    int tmp;
    cudaMemcpy(&tmp, d_ct_sim, sizeof(int), cudaMemcpyDeviceToHost);
    return tmp;
}

// Init PRNG on cpu. this is much slower.This is used in GATE and Micheala.Faster one see void wrap_init_PRNG_GPU
void wrap_init_PRNG_CPU(int seed, int block_size, int grid_size,
                              GPUParticleStack &h_g1, GPUParticleStack &h_g2,
                              GPUParticleStack &d_g1, GPUParticleStack &d_g2) {
    srand(seed);

    double time_start=get_time();
    int i=0; while (i<h_g1.size) {
        // init random seed
        h_g1.seed[i] = rand();
        h_g2.seed[i] = rand();
        ++i;
    }
    double time_used=get_time()-time_start;
    time_start=get_time();

    printf("the used time for loop random generation is: %f s\n", time_used);

    cudaMemcpy(d_g1.seed, h_g1.seed, sizeof(unsigned int)*h_g1.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g2.seed, h_g2.seed, sizeof(unsigned int)*h_g2.size, cudaMemcpyHostToDevice);

    double time_used2=get_time()-time_start;

        printf("the used time for copy is: %f s\n", time_used2);


//    // Kernel
    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;

    kernel_brent_init<<<grid, threads>>>(d_g1);
    kernel_brent_init<<<grid, threads>>>(d_g2);
    cudaThreadSynchronize();
}



// Init PRNG on GPU this is much faster then the wrap_init_PRNG_CPU() which is used in GATE
void wrap_init_PRNG(int seed, int block_size, int grid_size,
                              GPUParticleStack &h_g1, GPUParticleStack &h_g2,
                              GPUParticleStack &d_g1, GPUParticleStack &d_g2) {

	      curandGenerator_t gen1;
	     curandGenerator_t gen2;


        /* Create pseudo-random number generator */
        curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
        curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT);

        srand(seed);

        int seed1=rand();
        int seed2=rand();

        printf("seed1 is:  %d  seed2 is:%d \n",seed1,seed2);
        /* Set seed */
        curandSetPseudoRandomGeneratorSeed(gen1,seed1);
        curandSetPseudoRandomGeneratorSeed(gen2,seed2);

        /* Set seed */
//        curandSetGeneratorOffset(gen1,100);
//        curandSetGeneratorOffset(gen2,100);


        //Generate unsight int random number on device

        int DSIZE=d_g2.size;

//       curandGenerateUniform(gen, devData, DSIZE);
        curandGenerate(gen1, d_g1.seed, DSIZE);
//        cudaThreadSynchronize();
        /* Cleanup */
        curandDestroyGenerator(gen1);


         curandGenerate(gen2, d_g2.seed, DSIZE);
         cudaThreadSynchronize();

        curandDestroyGenerator(gen2);



//         printf("seed1 is:  %d  seed2 is:%d \n",d_g1.seed[10],d_g2.seed[10]);
//              printf("seed1 is:  %d  seed2 is:%d \n",seed1,seed2);


//    // Kernel
    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;

    kernel_brent_init<<<grid, threads>>>(d_g1);
    kernel_brent_init<<<grid, threads>>>(d_g2);
    cudaThreadSynchronize();
}






void wrap_init_PRNG_kernel_cuRand(int seed, int block_size, int grid_size,
                              GPUParticleStack &h_g1, GPUParticleStack &h_g2,
                              GPUParticleStack &d_g1, GPUParticleStack &d_g2) {



    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;

    int DSIZE=block_size*grid_size;

    curandState *devState;
    gpuErrchk(cudaMalloc((void**)&devState, DSIZE*sizeof(curandState)));
   cudaMalloc((void**)&devState, DSIZE*sizeof(curandState));
//    printf("size of curandstate: %d\n", sizeof(curandState));

   // Init rand seed for GPU
   kernel_random_seed_init<<<grid, threads>>>(d_g1,devState);
   kernel_random_seed_init<<<grid, threads>>>(d_g2,devState);

   cudaThreadSynchronize();

   cudaFree(devState);

    kernel_brent_init<<<grid, threads>>>(d_g1);
    kernel_brent_init<<<grid, threads>>>(d_g2);
    cudaThreadSynchronize();
}




// Voxelized source
void wrap_voxelized_source_b2b(GPUParticleStack &d_g1, GPUParticleStack &d_g2,
                               GPUPhantomActivities &d_act,
                               float3 phantom_size_in_mm,
                               int block_size, int grid_size) {
    

    // Kernel
    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;
    
    kernel_voxelized_source_b2b<<<grid, threads>>>(d_g1, d_g2, d_act, 
                                                   phantom_size_in_mm);
    cudaThreadSynchronize();
}

// Navigation
void wrap_navigation_regular(GPUParticleStack &d_g1, GPUParticleStack &d_g2,
                             GPUPhantom &phantom, GPUPhantomMaterials &materials, float radius,
                             int *d_ct_sim, int block_size, int grid_size) {
    // Kernel
    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;

    kernel_navigation_regular<<<grid, threads>>>(d_g1, phantom, materials, radius, d_ct_sim);
    kernel_navigation_regular<<<grid, threads>>>(d_g2, phantom, materials, radius, d_ct_sim);

    cudaThreadSynchronize();
}

// Detection
void wrap_detection(GPUParticleStack &d_g1, GPUParticleStack &d_g2,
                             GPUScanner &scanner, GPUPhantom &phantom,
                             int block_size, int grid_size) {
    // Kernel
    dim3 threads, grid;
    threads.x = block_size;
    grid.x = grid_size;

    kernel_detection_physics<<<grid, threads>>>(d_g1, scanner, phantom);
    kernel_detection_physics<<<grid, threads>>>(d_g2, scanner, phantom);

    cudaThreadSynchronize();
}


// Get back the gamma1 stack
void wrap_get_stack_gamma1(GPUParticleStack &h_g1, GPUParticleStack &d_g1) {
	int stack_size = h_g1.size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_stackpart_sint = stack_size * sizeof(short int);
    
	
	cudaMemcpy(h_g1.E,       d_g1.E, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.dx,      d_g1.dx, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.dy,      d_g1.dy, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.dz,      d_g1.dz, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.px,      d_g1.px, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.py,      d_g1.py, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.pz,      d_g1.pz, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_g1.tof,     d_g1.tof, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_g1.endsimu, d_g1.endsimu, mem_stackpart_char, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.active, d_g1.active, mem_stackpart_char, cudaMemcpyDeviceToHost);    
	cudaMemcpy(h_g1.crystalID, d_g1.crystalID, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.Edeposit, d_g1.Edeposit, mem_stackpart_float, cudaMemcpyDeviceToHost);    
	cudaMemcpy(h_g1.nCompton, d_g1.nCompton, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.nCompton_crystal, d_g1.nCompton_crystal, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g1.nPhotoelectric_crystal, d_g1.nPhotoelectric_crystal, mem_stackpart_sint, cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
}

// Get back the gamma1 stack
void wrap_get_stack_gamma2(GPUParticleStack &h_g2, GPUParticleStack &d_g2) {
	int stack_size = h_g2.size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_stackpart_sint = stack_size * sizeof(short int);    
	
	cudaMemcpy(h_g2.E,       d_g2.E, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.dx,      d_g2.dx, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.dy,      d_g2.dy, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.dz,      d_g2.dz, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.px,      d_g2.px, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.py,      d_g2.py, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.pz,      d_g2.pz, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_g2.tof,     d_g2.tof, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_g2.endsimu, d_g2.endsimu, mem_stackpart_char, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.active, d_g2.active, mem_stackpart_char, cudaMemcpyDeviceToHost);    
	cudaMemcpy(h_g2.crystalID, d_g2.crystalID, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.Edeposit, d_g2.Edeposit, mem_stackpart_float, cudaMemcpyDeviceToHost);    
	cudaMemcpy(h_g2.nCompton, d_g2.nCompton, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.nCompton_crystal, d_g2.nCompton_crystal, mem_stackpart_sint, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g2.nPhotoelectric_crystal, d_g2.nPhotoelectric_crystal, mem_stackpart_sint, cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
}

void wrap_free(GPUPhantomMaterials &d_mat, GPUPhantomActivities &d_act, 
               GPUPhantom &d_phan, GPUScanner &d_scanner, GPUParticleStack &d_g1, GPUParticleStack &d_g2, 
               GPUParticleStack &h_g1, GPUParticleStack &h_g2, int *d_ct_sim, int *d_ct_emit) {

    _materials_device_free(d_mat);
    _activities_device_free(d_act);
    _phantom_device_free(d_phan);
    _scanner_device_free(d_scanner);
    
    _stack_device_free(d_g1);
    _stack_device_free(d_g2);

    _stack_host_free(h_g1);
    _stack_host_free(h_g2);

    cudaFree(d_ct_sim);
    cudaFree(d_ct_emit);

    cudaThreadExit();

}
void wrap_free_counters(int *d_ct_sim, int *d_ct_emit) {

    cudaFree(d_ct_sim);
    cudaFree(d_ct_emit);
}


//#endif
