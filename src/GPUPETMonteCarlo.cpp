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

#ifndef GPUPETLIB_CPP
#define GPUPETLIB_CPP

#include "../inc/GPUPETMonteCarlo.h"

//// GPUPETMonteCarlo class //////////////////////////////////////////////////

GPUPETMonteCarlo::GPUPETMonteCarlo()
:m_stack_size(0),
m_nb_of_particles(0),
m_block_size(0),
m_grid_size(0),

d_materials(),
d_activities(),
d_phantom(),

d_scanner(),

d_gamma1(),
d_gamma2(),
h_gamma1(),
h_gamma2(),


d_ct_sim(NULL),
d_ct_emit(NULL),
h_ct_sim(0),
h_ct_emit(0),
h_time(0)
{


}

// Set device
void GPUPETMonteCarlo::set_device(int id) {
    wrap_set_device(id);
}

// Copy materials to device
void GPUPETMonteCarlo::copy_materials_to_device(GPUPhantomMaterials h_materials) {
    wrap_copy_materials_to_device(h_materials, d_materials);
}

// Copy phantom to device
void GPUPETMonteCarlo::copy_phantom_to_device(GPUPhantom h_phantom) {
    wrap_copy_phantom_to_device(h_phantom, d_phantom);
}

// Copy activities to device
void GPUPETMonteCarlo::copy_activities_to_device(GPUPhantomActivities h_act) {
    wrap_copy_activities_to_device(h_act, d_activities);
}

// Copy scanner to device
void GPUPETMonteCarlo::copy_scanner_to_device(GPUScanner h_scan) {
    wrap_copy_scanner_to_device(h_scan, d_scanner);
}
    
// Init particle stack
void GPUPETMonteCarlo::init_particle_stack(int size) {
    wrap_init_particle_stack(size, h_gamma1, h_gamma2, d_gamma1, d_gamma2);
    m_stack_size = size;
}

// Init simulation
void GPUPETMonteCarlo::init_particle_counter() {
    h_ct_sim = 0.0f;
    h_ct_emit = 0.0f;
    d_ct_sim = wrap_init_counter(h_ct_sim);
    d_ct_emit = wrap_init_counter(h_ct_emit);
}

// Set time
void GPUPETMonteCarlo::set_time(float time) {
    h_time = time;
}

// Set the number of particles required
void GPUPETMonteCarlo::set_nb_of_particles(int nb) {
    m_nb_of_particles = nb;
}

// Init PRNG
void GPUPETMonteCarlo::init_PRNG(int seed) {
    wrap_init_PRNG(seed, m_block_size, m_grid_size, h_gamma1, h_gamma2, d_gamma1, d_gamma2);
}

// Set grid the grid of threads
void GPUPETMonteCarlo::set_grid_device(int block_size) {
    m_block_size = block_size;
    m_grid_size = (m_stack_size + m_block_size - 1) / m_block_size;
}
        
// Get back the stack gamma1
GPUParticleStack GPUPETMonteCarlo::get_stack_gamma1() {
    wrap_get_stack_gamma1(h_gamma1, d_gamma1);
    return h_gamma1;
}

// Get back the stack gamma2
GPUParticleStack GPUPETMonteCarlo::get_stack_gamma2() {
    wrap_get_stack_gamma2(h_gamma2, d_gamma2);
    return h_gamma2;
}
        
// Get back the number of particles required
int GPUPETMonteCarlo::get_nb_of_particles() {
    return m_nb_of_particles;
}

// Get back the number of simulated particles
int GPUPETMonteCarlo::get_nb_of_simulated() {
    return wrap_get_nb_of_simulated(d_ct_sim);
}

// Voxelized source (back2back)
void GPUPETMonteCarlo::voxelized_source_b2b() {
    wrap_voxelized_source_b2b(d_gamma1, d_gamma2, d_activities,
                              d_phantom.size_in_mm,
                              m_block_size, m_grid_size);
}

// Navigation
void GPUPETMonteCarlo::navigation() {
    wrap_navigation_regular(d_gamma1, d_gamma2, d_phantom, d_materials, d_scanner.cyl_radius, d_ct_sim,
                            m_block_size, m_grid_size);
}
// Navigation
void GPUPETMonteCarlo::detection() {
    wrap_detection(d_gamma1, d_gamma2, d_scanner, d_phantom,
                            m_block_size, m_grid_size);
}
// Free particle counters
void GPUPETMonteCarlo::free_counters() {
    wrap_free_counters(d_ct_sim, d_ct_emit);
}
// Free memory and device
void GPUPETMonteCarlo::free() {
    wrap_free(d_materials, d_activities, d_phantom, d_scanner, d_gamma1, d_gamma2, 
              h_gamma1, h_gamma2, d_ct_sim, d_ct_emit);
}

GPUPETMonteCarlo::~GPUPETMonteCarlo()
{
	free();

}

#endif
