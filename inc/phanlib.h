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

#ifndef PHANLIB_H
#define PHANLIB_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include "structure.h"
#include "MaterialDataBase.h"




///// Phantom management /////////////////////////////////////////////////////////////

class Phantom {
    public:
        Phantom();
        void load_from_raw(std::string filename, int nx, int ny, int nz,
				 			std::string filename_att, int nx_att, int ny_att, int nz_att,
                                                 float sx, float sy, float sz);
        void label_from_range(std::string filename);
        void activity_from_range(std::string filename);
		void activity_from_image(float);

        GPUPhantom get_phantom_for_GPU(); 
        GPUPhantomActivities get_activities_for_GPU();
        GPUPhantomMaterials get_materials_for_GPU(MaterialDataBase);

    private:
        float *m_raw_data;
		float *m_atten_data;
        unsigned short int m_nx, m_ny, m_nz, m_nx_att, m_ny_att, m_nz_att;
        unsigned int m_nb_voxels, m_nb_voxels_att;
        unsigned int m_mem_size, m_mem_size_att;
        float m_spacing_x, m_spacing_y, m_spacing_z;
        
        unsigned short int *m_lbl_data; // labeled data
        float *m_act_data;              // activity data
        std::vector<std::string> m_list_of_materials;

        float read_start_range(std::string);
        float read_stop_range(std::string);
        float read_act_range(std::string);
        std::string read_mat_range(std::string);
        void skip_comment(std::istream &);

};

#endif
