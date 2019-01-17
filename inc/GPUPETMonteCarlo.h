/*
 * GPUPETMonteCarlo.h
 *
 *  Created on: Jul 20, 2018
 *      Author: bma
 */

#ifndef _GPUPETMONTECARLO_H_
#define _GPUPETMONTECARLO_H_

#include <stdlib.h>
#include <stdio.h>
#include "structure_device.h"
#include "gpu_common_func.cuh"

//CUDA RunTime API
#include <cuda_runtime.h>



class GPUPETMonteCarlo {
    public:
        GPUPETMonteCarlo();
        void copy_materials_to_device(GPUPhantomMaterials);
        void copy_phantom_to_device(GPUPhantom);
        void copy_activities_to_device(GPUPhantomActivities);
        void copy_scanner_to_device(GPUScanner);

        void init_particle_stack(int);
        void init_particle_counter();
        void init_PRNG(int);

        void set_nb_of_particles(int nb);
        void set_device(int);
        void set_grid_device(int);
        void set_time(float);

        GPUParticleStack get_stack_gamma1();
        GPUParticleStack get_stack_gamma2();
        int get_nb_of_particles();
        int get_nb_of_simulated();

        void voxelized_source_b2b();
        void navigation();
        void detection();

        void free_counters();
        void free();
        ~GPUPETMonteCarlo();

    private:
        int m_stack_size;
        int m_nb_of_particles;
        int m_block_size;
        int m_grid_size;

        // phantom
        GPUPhantomMaterials d_materials;
        GPUPhantomActivities d_activities;
        GPUPhantom d_phantom;

        //scanner
        GPUScanner d_scanner;

        // particle stack
        GPUParticleStack d_gamma1, d_gamma2;
        GPUParticleStack h_gamma1, h_gamma2;

        // to count simulated and emitted particle
        int *d_ct_sim, *d_ct_emit;
        int h_ct_sim, h_ct_emit;

        // time
        float h_time;
};




#endif /* GPUPETMONTECARLO_H_ */
