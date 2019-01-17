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

#ifndef PHANLIB_CPP
#define PHANLIB_CPP


#include "../inc/phanlib.h"

using namespace std;

//// Phantom class ////////////////////////////////////////////////////////

Phantom::Phantom()
:m_raw_data(NULL),
m_atten_data(NULL),
 m_nx(0),
m_ny(0),
m_nz(0),
m_nx_att(0),
m_ny_att(0),
m_nz_att(0),

 m_nb_voxels(0),
m_nb_voxels_att(0),
 m_mem_size(0),
m_mem_size_att(0),
m_spacing_x(0),
m_spacing_y(0),
m_spacing_z(0),

m_lbl_data(NULL), // labeled data
m_act_data(NULL),
// activity data
m_list_of_materials()
{
}

// Load phantom from binary data (float)
void Phantom::load_from_raw(std::string filename, int nx, int ny, int nz,
							std::string filename_atten, int nx_att, int ny_att, int nz_att,
                                                  float sx, float sy, float sz) {
	long size;
    FILE *pfile = fopen(filename.c_str(), "rb");
	if (pfile==NULL){
    	printf("Activity File could not be opened: %s\n", filename.c_str());
		exit(0);
	}
	else{
		fseek(pfile, 0, SEEK_END);
    	size=ftell(pfile);
		fseek(pfile, 0, SEEK_SET);
		if(size!=nx*ny*nz*4){
    		printf("Activity File has wrong size %s: %li = %i\n", filename.c_str(), size,nx*ny*nz*4 );
			exit(0);
		}
	}

    FILE *afile = fopen(filename_atten.c_str(), "rb");
	if (afile==NULL){
    	printf("Atten File could not be opened: %s\n", filename_atten.c_str());
		exit(0);
	}
	else{
		fseek(afile, 0, SEEK_END);
    	size=ftell(afile);
		fseek(afile, 0, SEEK_SET);
		if(size!=nx_att*ny_att*nz_att*4){
    		printf("Atten File has wrong size %s: %li != %i\n", filename_atten.c_str(), size,nx_att*ny_att*nz_att*4 );
			exit(0);
		}
	}

    m_nb_voxels = nx*ny*nz;
    m_nx = nx;
    m_ny = ny;
    m_nz = nz;
	m_nb_voxels_att= nx_att*ny_att*nz_att;
	m_nx_att=nx_att;
	m_ny_att=ny_att;
	m_nz_att=nz_att;
    m_spacing_x = sx;
    m_spacing_y = sy;
    m_spacing_z = sz;
    m_mem_size = sizeof(float)*m_nb_voxels;
    m_mem_size_att = sizeof(float)*m_nb_voxels_att;

    m_raw_data = (float*)malloc(m_mem_size);
    m_atten_data= (float*)malloc(m_mem_size_att);

    int file_size_em=fread(m_raw_data, sizeof(float), m_nb_voxels, pfile);
    if(file_size_em!=m_nb_voxels)
    {
    	printf("emission image saved wrong: file size: %d \n", file_size_em);
    	exit(0);

    }

    int file_size_att= fread(m_atten_data, sizeof(float), m_nb_voxels_att, afile);
    if(file_size_att!=m_nb_voxels_att)
        {

        printf("attenuation image saved wrong: file size: %d \n", file_size_att);
        exit(0);

        }

    fclose(pfile);
    fclose(afile);
}

// Skip comment starting with "#"
void Phantom::skip_comment(std::istream & is) {
    char c;
    char line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Read start range
float Phantom::read_start_range(std::string txt) {
    float res;
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read stop range
float Phantom::read_stop_range(std::string txt) {
    float res;
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read material range
std::string Phantom::read_mat_range(std::string txt) {
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(txt.find(" ")+1);
    return txt.substr(0, txt.find(" "));
}

// Read activity range
float Phantom::read_act_range(std::string txt) {
    float res;
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Labeled phantom according range materials definition
void Phantom::label_from_range(std::string filename) {

    float start, stop;
    std::string mat_name, line;
    int i;
    float val;
    unsigned short int mat_index=0;

    printf("mateiral file path is : %s\n", filename.c_str());
    // allocate lbl_data
    m_lbl_data = (unsigned short int*)malloc(sizeof(unsigned short int) * m_nb_voxels_att);

    // Read range file
    std::ifstream file(filename.c_str());

	if(file.fail()){
		printf("Error while reading range file %s\n", filename.c_str());
		exit(0);
	}
    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {

            start = read_start_range(line);
            stop  = read_stop_range(line);
            mat_name = read_mat_range(line);

            printf("start is: %f  stop is : %f \n", start,stop);
            printf("mateiral name is: %s\n", mat_name.c_str());
            m_list_of_materials.push_back(mat_name);

            // build labeled phantom according range data
            i=0; while (i < m_nb_voxels_att) {
                val = m_atten_data[i];

                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    m_lbl_data[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;
        
    } // read file

}

// Activity phantom according range activity definition
void Phantom::activity_from_range(std::string filename) {

    float start, stop, act, val;
    int i;
    std::string line;

    // allocate and set activity data to zeros
    m_act_data = (float*)malloc(sizeof(float) * m_nb_voxels);
    i=0; while(i<m_nb_voxels) {m_act_data[i]=0.0f; ++i;}

    // Read range file
    std::ifstream file(filename.c_str());
    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = read_start_range(line);
            stop  = read_stop_range(line);
            act   = read_act_range(line);

            // build labeled phantom according range data
            i=0; while (i < m_nb_voxels) {
                val = m_raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    m_act_data[i] = act;
                }
                ++i;
            } // over the volume

        } // new material range
        
    } // read file

}
// Activity phantom according to image values, values under threshold are not used
void Phantom::activity_from_image(float threshold) {

    // allocate and set activity data to zeros
    m_act_data = (float*)malloc(sizeof(float) * m_nb_voxels);

	float val;
    int i=0; 
	while(i<m_nb_voxels) {
		m_act_data[i]=0.0f;
		val = m_raw_data[i];
	  	if (val>threshold) {
			m_act_data[i] = val;
		}	
	   	++i;
	}
}

// Return phantom data for the GPU
GPUPhantom Phantom::get_phantom_for_GPU() {
    GPUPhantom phan;
    
    phan.data = m_lbl_data;
    phan.mem_data = m_nb_voxels_att * sizeof(unsigned short int);
    phan.nb_voxel_slice = m_nx_att*m_ny_att;
    phan.nb_voxel_volume = phan.nb_voxel_slice * m_nz_att;

    phan.voxel_size.x = m_spacing_x;
    phan.voxel_size.y = m_spacing_y;
    phan.voxel_size.z = m_spacing_z;
    
    phan.size_in_vox.x = m_nx_att;
    phan.size_in_vox.y = m_ny_att;
    phan.size_in_vox.z = m_nz_att;
    
    phan.size_in_mm.x = m_nx_att*m_spacing_x;
    phan.size_in_mm.y = m_ny_att*m_spacing_y;
    phan.size_in_mm.z = m_nz_att*m_spacing_z;

    return phan;
}

// Compute and return CDF activity for the GPU
GPUPhantomActivities Phantom::get_activities_for_GPU() {
    GPUPhantomActivities act;

	act.voxel_size.x = m_spacing_x;
    act.voxel_size.y = m_spacing_y;
    act.voxel_size.z = m_spacing_z;
    
    act.size_in_vox.x = m_nx;
    act.size_in_vox.y = m_ny;
    act.size_in_vox.z = m_nz;
    
    act.size_in_mm.x = m_nx*m_spacing_x;
    act.size_in_mm.y = m_ny*m_spacing_y;
    act.size_in_mm.z = m_nz*m_spacing_z;

    // count nb of non zeros activities
    int i;
    unsigned int nb=0;
    i=0; while (i<m_nb_voxels) {
        if (m_act_data[i] != 0.0f) ++nb;
        ++i;
    }
    act.nb_activities = nb;
	printf("Nb of act: %i \n", nb);
    // mem allocation
    act.act_index = (unsigned int*)malloc(nb*sizeof(unsigned int));
    act.act_cdf = (float*)malloc(nb*sizeof(float));
    double* cdf=new double[nb];

    // fill array with non zeros values activity
    int index = 0;
    float val;
    double sum = 0.0; // for the cdf
    i=0; while (i<m_nb_voxels) {
        val = m_act_data[i];
        if (val != 0.0f) {
            act.act_index[index] = i;
            cdf[index] = val;
            sum += val;
            ++index;
        }
        ++i;
    }
    act.tot_activity = (float)sum;

    // compute cummulative density function
    cdf[0] /= sum;
    act.act_cdf[0]=cdf[0];
    i=1; 
    while (i<nb) {
        cdf[i] = (cdf[i]/sum) + cdf[i-1];
        act.act_cdf[i]= (float) cdf[i];
        ++i;
    }
    // Watchdog FIXME why the last one is not zeros (float/double error)
    act.act_cdf[nb-1] = 1.0f;

    delete cdf;

    return act;
}

// Compute and return materials for the GPU
GPUPhantomMaterials Phantom::get_materials_for_GPU(MaterialDataBase db) {

    GPUPhantomMaterials gpumat;
    
    // nb of materials
    gpumat.nb_materials = m_list_of_materials.size();
    gpumat.nb_elements = (unsigned short int*)malloc(sizeof(unsigned short int) 
                                                        * gpumat.nb_materials);
    gpumat.index = (unsigned short int*)malloc(sizeof(unsigned short int) 
                                                        * gpumat.nb_materials);

    int i, j;
    unsigned int access_index = 0;
    unsigned int fill_index = 0;
    std::string mat_name, elt_name;
    Material cur_mat;

    i=0; while (i < gpumat.nb_materials) {
        // get mat name
        mat_name = m_list_of_materials[i];

        // read mat from databse
        cur_mat = db.materials_database[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database\n", mat_name.c_str());
            exit(EXIT_FAILURE);
        }

        // get nb of elements
        gpumat.nb_elements[i] = cur_mat.nb_elements;

        // compute index
        gpumat.index[i] = access_index;
        access_index += cur_mat.nb_elements;
        
        ++i;
    }

    // nb of total elements
    gpumat.nb_elements_total = access_index;
    gpumat.mixture = (unsigned short int*)malloc(sizeof(unsigned short int)*access_index);
    gpumat.atom_num_dens = (float*)malloc(sizeof(float)*access_index);

    // store mixture element and compute atomic density
    i=0; while (i < gpumat.nb_materials) {
        // get mat name
        mat_name = m_list_of_materials[i];
        // read mat from databse
        cur_mat = db.materials_database[mat_name];

        j=0; while (j < cur_mat.nb_elements) {
            // read element name    
            elt_name = cur_mat.mixture_Z[j];

            // store Z
            gpumat.mixture[fill_index] = db.elements_Z[elt_name];

            // compute atom num dens (Avo*fraction*dens) / Az
            gpumat.atom_num_dens[fill_index] = Avogadro/db.elements_A[elt_name] * 
                                               cur_mat.mixture_f[j]*cur_mat.density; 

            ++j;
            ++fill_index;
        }
        ++i;
    }

    return gpumat;
}



#endif
