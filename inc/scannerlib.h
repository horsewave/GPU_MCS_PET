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

#ifndef SCANNERLIB_H
#define SCANNERLIB_H

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <list>
#include "phanlib.h"

//#include "gpupetlib.cuh"


#include "structure.h"


//#ifndef my_float3
//#define my_float3
//struct my_float3 {
//    float x, y, z;
//};
//#endif
//
//#ifndef GPUSCANNER
//#define GPUSCANNER
//// GPU SoA for scanner geometry
//struct GPUScanner{
//    float cyl_radius;
//    float cyl_halfheight;
//    float block_pitch;
//    float cta_pitch;
//    float cax_pitch;
//    unsigned short int ncass;
//    unsigned short int nblock;
//    unsigned short int ncry_ta;
//    unsigned short int ncry_ax;
//    unsigned short int blocksize;
//    my_float3 halfsize;
//    my_float3* pos;
//    my_float3* v0;
//    my_float3* v1;
//    my_float3* v2;
//    GPUPhantomMaterials* mat;
//};
//#endif
//
//
//typedef struct {
//    double time;
//    int id;
//    int eventID;
//    short int nCompton;
//}Single ;
//
//
//typedef struct {
//    Single one;
//    Single two;
//    char type;
//}Coincidence;

class Scanner {
    public:
        Scanner(float,float,float,float,float,
                unsigned short int, unsigned short int, unsigned short int, unsigned short int,
                float,float,float, std::string, float, float);

        GPUPhantomMaterials defineMaterial(MaterialDataBase);

        GPUScanner get_scanner_for_GPU(MaterialDataBase);
        std::list<Single> create_singles_list(GPUParticleStack, GPUParticleStack, float );
        void process_to_singles(GPUParticleStack, GPUParticleStack, std::list<Single>*, float );

        //Get the photoelectric and compton scatter times in crystal
        void Get_hits_info(GPUParticleStack, GPUParticleStack );
        void save_coincidences(std::list<Single>*, std::string, bool, bool , float* lorvec_scat=NULL, float* lorvec_true=NULL);
        void save_coincidences_lor(std::list<Single>*, std::string, bool, bool , float* lortrue=NULL, float* lorscat=NULL);

		// function for LORfileData lookup (Juelich format)
		void create_hplookup(short int *hplookup);

		// function for LORfileData lookup (Juelich format)
		int get_LOR_flat_address(short int *hplookup, int h1, int b1, int c1,int h2, int b2,int c2);

        ~Scanner();

    private:
        float radius;
        float halfheight;
        float block_pitch;
        float cta_pitch;
        float cax_pitch;
        unsigned short int ncass;
        unsigned short int nblock;
        unsigned short int ncry_ta;
        unsigned short int ncry_ax;
        int blocksize;
        int casssize;
        my_float3 halfsize;
        my_float3* pos;
        my_float3* v0;
        my_float3* v1;
        my_float3* v2;
        std::string mat_name;
        double coinc_window;
        float lld;

};

#endif

