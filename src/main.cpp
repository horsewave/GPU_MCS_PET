#include <stdio.h>
#include <iostream>

#include "../inc/run_mcs.h"

using namespace std;

void Mcs_simu(string patient_name, string frame_time, string inputFolder,
		int nr_of_runs) {

	string basename = patient_name + "_" + frame_time;
	string basename_simu = basename + "_gpuSimu";


  string emission_image_ori=inputFolder+"img/sss/GUI_generated/range_"+frame_time+"_reco_W3_na.dat";////emission image
  string atten_image_head_ori =inputFolder +"atten/" +"XB1BN286N_AttenMap_HeadOnly.i";
	string atten_image_coils =
			"/localdata/Ma/software/src/gpupet/DATA/ctJuelich_Komplett_rsl_segmented_320x153_nobed.i";

	string simufolder = inputFolder + "scatterMCS/";

	string emission_image_pro = simufolder + basename_simu + "_activity.i";
	string atten_image_pro = simufolder + basename_simu + "_atten.i";

	string root_simu_log = simufolder + "mcs.log";

	ofstream of;

	of.open(root_simu_log.c_str(), std::ofstream::out | std::ofstream::app);

//  bool is_use_coil =true;
	bool is_use_coil = false;
	int nr_of_gpus = 2;
//  int block_size=128;
	int block_size = 96;
//  int nr_of_runs=2000;//24*E9
	int photon_num_per_run = 6000000;

	int dim_x = 256;
	int dim_y = 256;
	int dim_z = 153;

	float voxel_size = 1.25;

	//---run GPU simulation/////////////////////////////////////////////////////////////////////////////////////////
		string path_element =
				"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/elts.dat";
		string path_material =
				"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/mats.dat";
		string path_rang =
				"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/pet_data/range_brain.dat";

		cout << "start of simulation" << endl;

	///.....path for the simulated data
	string fname_simu_run = simufolder + basename_simu;
	//string fname_simu_log = simufolder + basename_simu + "_" + nb.str() + ".log";
	cout << "file name input:  " << fname_simu_run << endl;
	cout << "number of GPU:  " << nr_of_gpus << endl;
	cout << "number of runs:  " << nr_of_runs << endl;
	cout << "photons:  " << photon_num_per_run << endl;

	Run_mcs *Run_mcs_simulation = new Run_mcs;

	cout << "new class Run_mcs  " << endl;

	Run_mcs_simulation->Set_file_name(fname_simu_run);
	cout << "new class Run_mcs->set_file_name" << endl;

	Run_mcs_simulation->Set_material_path(path_element, path_material);
	cout << "new class Run_mcs->Set_material_path  " << endl;
	Run_mcs_simulation->Set_phantom_path(path_rang);
	cout << "new class Run_mcs->Set_phantom_path  " << endl;
	Run_mcs_simulation->Set_path_emi_img(emission_image_ori,
			emission_image_pro);
	cout << "new class Run_mcs->Set_path_emi_img  " << endl;
	Run_mcs_simulation->Set_path_att_img(atten_image_head_ori, atten_image_pro,
			atten_image_coils);
	cout << "new class Run_mcs->Set_path_att_img  " << endl;
	Run_mcs_simulation->Set_stack_size(photon_num_per_run);
	cout << "new class Run_mcs->Set_stack_size  " << endl;
	Run_mcs_simulation->Set_GPU_device_num(nr_of_gpus);
	;
	cout << "new class Run_mcs->Set_GPU_device_num  " << endl;
	Run_mcs_simulation->Set_coil_flag(is_use_coil);
	cout << "new class Run_mcs->Set_coil_flag  " << endl;
	Run_mcs_simulation->Set_num_runs(nr_of_runs);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Set_image_dim_no_coil(dim_x, dim_y, dim_z);
	//  Run_mcs_simulation->Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z);
	Run_mcs_simulation->Set_image_voxel_size(voxel_size);

	Run_mcs_simulation->Set_thread_block_size(block_size);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Run_mcs_process();

	cout << "new class Run_mcs->Run_mcs_process  " << endl;

	cout << "end of simulation" << endl;

	of.close();

	if (Run_mcs_simulation != NULL) {
		delete Run_mcs_simulation;
		Run_mcs_simulation = NULL;

	}

}

void Mcs_simu_brain_phantom(string patient_name, string frame_time,
		string inputFolder, int nr_of_runs) {

	string basename = patient_name;
	string basename_simu = basename + "_gpuSimu";

	//input images
	string emission_image_ori = inputFolder
			+ "Brain_phantom_source_210_float.i33"; ////emission image
	string atten_image_head_ori = inputFolder
			+ "Brain_phantom_att_210_float.i33";
	string atten_image_coils =
			"/localdata/Ma/software/src/gpupet/DATA/ctJuelich_Komplett_rsl_segmented_320x153_nobed.i";

	string simufolder = inputFolder + "scatterMCS/";

	string emission_image_pro = simufolder + basename_simu + "_activity.i";
	string atten_image_pro = simufolder + basename_simu + "_atten.i";

	string root_simu_log = simufolder + "mcs.log";

	ofstream of;

	of.open(root_simu_log.c_str(), std::ofstream::out | std::ofstream::app);

	bool is_use_coil = false;
	int nr_of_gpus = 2;
	int block_size = 96;
//  int nr_of_runs=2000;//24*E9
//  int photon_num_per_run=6000000;
	int photon_num_per_run = 1000000;

	int dim_x = 210;
	int dim_y = 210;
	int dim_z = 153;
//  float voxel_size=1.25;
	float voxel_size = 1.25;

	//---run GPU simulation/////////////////////////////////////////////////////////////////////////////////////////
	string path_element =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/elts.dat";
	string path_material =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/mats.dat";
	string path_rang =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/pet_data/range_brain.dat";

	cout << "start of simulation" << endl;

	///.....path for the simulated data
	string fname_simu_run = simufolder + basename_simu;
	//string fname_simu_log = simufolder + basename_simu + "_" + nb.str() + ".log";
	cout << "file name input:  " << fname_simu_run << endl;
	cout << "number of GPU:  " << nr_of_gpus << endl;
	cout << "number of runs:  " << nr_of_runs << endl;
	cout << "photons:  " << photon_num_per_run << endl;

	Run_mcs *Run_mcs_simulation = new Run_mcs;

	cout << "new class Run_mcs  " << endl;

	Run_mcs_simulation->Set_file_name(fname_simu_run);
	cout << "new class Run_mcs->set_file_name" << endl;

	Run_mcs_simulation->Set_material_path(path_element, path_material);
	cout << "new class Run_mcs->Set_material_path  " << endl;
	Run_mcs_simulation->Set_phantom_path(path_rang);
	cout << "new class Run_mcs->Set_phantom_path  " << endl;
	Run_mcs_simulation->Set_path_emi_img(emission_image_ori,
			emission_image_pro);
	cout << "new class Run_mcs->Set_path_emi_img  " << endl;
	Run_mcs_simulation->Set_path_att_img(atten_image_head_ori, atten_image_pro,
			atten_image_coils);
	cout << "new class Run_mcs->Set_path_att_img  " << endl;
	Run_mcs_simulation->Set_stack_size(photon_num_per_run);
	cout << "new class Run_mcs->Set_stack_size  " << endl;
	Run_mcs_simulation->Set_GPU_device_num(nr_of_gpus);
	;
	cout << "new class Run_mcs->Set_GPU_device_num  " << endl;
	Run_mcs_simulation->Set_coil_flag(is_use_coil);
	cout << "new class Run_mcs->Set_coil_flag  " << endl;

	Run_mcs_simulation->Set_image_dim_no_coil(dim_x, dim_y, dim_z);
//  Run_mcs_simulation->Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z);
	Run_mcs_simulation->Set_image_voxel_size(voxel_size);

	Run_mcs_simulation->Set_num_runs(nr_of_runs);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Set_thread_block_size(block_size);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Run_mcs_process();

	cout << "new class Run_mcs->Run_mcs_process  " << endl;

	cout << "end of simulation" << endl;

	of.close();

	if (Run_mcs_simulation != NULL) {
		delete Run_mcs_simulation;
		Run_mcs_simulation = NULL;

	}

}

void Mcs_simu_cubic_phantom(string patient_name, string frame_time,
		string inputFolder, int nr_of_runs) {

	string basename = patient_name;
	string basename_simu = basename + "_gpuSimu";

	//input images
	string emission_image_ori = inputFolder + "cubic_phantom_source.i33"; ////emission image
	string atten_image_head_ori = inputFolder + "cubic_phantom_att.i33";
//  string emission_image_ori=inputFolder+"cubic_phantom_source_large.i33";////emission image
//    string atten_image_head_ori =inputFolder +"cubic_phantom_att_large.i33";
	string atten_image_coils =
			"/localdata/Ma/software/src/gpupet/DATA/ctJuelich_Komplett_rsl_segmented_320x153_nobed.i";

	string simufolder = inputFolder + "scatterMCS/";

	string emission_image_pro = simufolder + basename_simu + "_activity.i";
	string atten_image_pro = simufolder + basename_simu + "_atten.i";

	string root_simu_log = simufolder + "mcs.log";
	ofstream of;

	of.open(root_simu_log.c_str(), std::ofstream::out | std::ofstream::app);

	bool is_use_coil = false;
//  int nr_of_gpus=2;
	int nr_of_gpus = 1;
//  int block_size=320;
	int block_size = 96;
//  int nr_of_runs=2000;//24*E9
//  int photon_num_per_run=6000000;
	int photon_num_per_run = 1000000;

	int dim_x = 100;
	int dim_y = 100;
	int dim_z = 100;
//
//  int dim_x=256;
//    int dim_y=256;
//    int dim_z=153;
	float voxel_size = 1.25;
//  float voxel_size=1.00;

//---run GPU simulation/////////////////////////////////////////////////////////////////////////////////////////
	string path_element =
			"/data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/elts.dat";
	string path_material =
			"/data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/mats.dat";
	string path_rang =
			"/data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/range_brain.dat";

	cout << "start of simulation" << endl;

	///.....path for the simulated data;
	string fname_simu_run = simufolder + basename_simu;
	//string fname_simu_log = simufolder + basename_simu + "_" + nb.str() + ".log";
	cout << "file name input:  " << fname_simu_run << endl;
	cout << "number of GPU:  " << nr_of_gpus << endl;
	cout << "number of runs:  " << nr_of_runs << endl;
	cout << "photons:  " << photon_num_per_run << endl;

	Run_mcs *Run_mcs_simulation = new Run_mcs;

	cout << "new class Run_mcs  " << endl;

	Run_mcs_simulation->Set_file_name(fname_simu_run);
	cout << "new class Run_mcs->set_file_name" << endl;

	Run_mcs_simulation->Set_material_path(path_element, path_material);
	cout << "new class Run_mcs->Set_material_path  " << endl;
	Run_mcs_simulation->Set_phantom_path(path_rang);
	cout << "new class Run_mcs->Set_phantom_path  " << endl;
	Run_mcs_simulation->Set_path_emi_img(emission_image_ori,
			emission_image_pro);
	cout << "new class Run_mcs->Set_path_emi_img  " << endl;
	Run_mcs_simulation->Set_path_att_img(atten_image_head_ori, atten_image_pro,
			atten_image_coils);
	cout << "new class Run_mcs->Set_path_att_img  " << endl;
	Run_mcs_simulation->Set_stack_size(photon_num_per_run);
	cout << "new class Run_mcs->Set_stack_size  " << endl;
	Run_mcs_simulation->Set_GPU_device_num(nr_of_gpus);
	;
	cout << "new class Run_mcs->Set_GPU_device_num  " << endl;
	Run_mcs_simulation->Set_coil_flag(is_use_coil);
	cout << "new class Run_mcs->Set_coil_flag  " << endl;

	Run_mcs_simulation->Set_image_dim_no_coil(dim_x, dim_y, dim_z);
//  Run_mcs_simulation->Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z);
	Run_mcs_simulation->Set_image_voxel_size(voxel_size);

	Run_mcs_simulation->Set_num_runs(nr_of_runs);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Set_thread_block_size(block_size);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Run_mcs_process();

	cout << "new class Run_mcs->Run_mcs_process  " << endl;

	cout << "end of simulation" << endl;

	of.close();

	if (Run_mcs_simulation != NULL) {
		delete Run_mcs_simulation;
		Run_mcs_simulation = NULL;

	}

}

void Mcs_simu_ofov() {

	string basename = "test";
	string basename_simu = basename + "_gpuSimu";

	string inputFolder =
			"/data/PET/mr_pet_temp/Ma/software/data/gpupet/ofov/test/";
	//input images
	string emission_image_ori = inputFolder + "emission_merged.data"; ////emission image
	string atten_image_head_ori = inputFolder + "att_map_merged.data";

	string atten_image_coils = inputFolder
			+ "att_map_merged_coil_320_320_306.data";

	string simufolder =
			"/data/PET/mr_pet_temp/Ma/software/data/gpupet/ofov/test/scatterMCS/";

	string emission_image_pro = simufolder + "activity.i";
	string atten_image_pro = simufolder + "atten.i";

	string root_simu_log = simufolder + "mcs.log";
	ofstream of;

	of.open(root_simu_log.c_str(), std::ofstream::out | std::ofstream::app);

	bool is_use_coil = false;

	int nr_of_gpus = 2;
//  int block_size=320;
	int block_size = 96;
//  int nr_of_runs=2000;//24*E9
	int nr_of_runs = 8; //24*E9
	int photon_num_per_run = 8000000;
//  int photon_num_per_run=1000000;

//
	int dim_x = 256;
	int dim_y = 256;
	int dim_z = 306;
	float voxel_size = 1.25;
//  float voxel_size=1.00;

//---run GPU simulation/////////////////////////////////////////////////////////////////////////////////////////
	string path_element =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/elts.dat";
	string path_material =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse/data/mats.dat";
	string path_rang =
			"/data/PET/mr_pet_temp/Ma/software/src/cuda/gpu_mcs/gpu_mcs_eclipse_ofov/pet_data/range_brain.dat";

	cout << "start of simulation" << endl;

	///.....path for the simulated data;
	string fname_simu_run = simufolder + basename_simu;
	//string fname_simu_log = simufolder + basename_simu + "_" + nb.str() + ".log";
	cout << "file name input:  " << fname_simu_run << endl;
	cout << "number of GPU:  " << nr_of_gpus << endl;
	cout << "number of runs:  " << nr_of_runs << endl;
	cout << "photons:  " << photon_num_per_run << endl;

	Run_mcs *Run_mcs_simulation = new Run_mcs;

	cout << "new class Run_mcs  " << endl;

	Run_mcs_simulation->Set_file_name(fname_simu_run);
	cout << "new class Run_mcs->set_file_name" << endl;

	Run_mcs_simulation->Set_material_path(path_element, path_material);
	cout << "new class Run_mcs->Set_material_path  " << endl;
	Run_mcs_simulation->Set_phantom_path(path_rang);
	cout << "new class Run_mcs->Set_phantom_path  " << endl;
	Run_mcs_simulation->Set_path_emi_img(emission_image_ori,
			emission_image_pro);
	cout << "new class Run_mcs->Set_path_emi_img  " << endl;
	Run_mcs_simulation->Set_path_att_img(atten_image_head_ori, atten_image_pro,
			atten_image_coils);
	cout << "new class Run_mcs->Set_path_att_img  " << endl;
	Run_mcs_simulation->Set_stack_size(photon_num_per_run);
	cout << "new class Run_mcs->Set_stack_size  " << endl;
	Run_mcs_simulation->Set_GPU_device_num(nr_of_gpus);
	;
	cout << "new class Run_mcs->Set_GPU_device_num  " << endl;
	Run_mcs_simulation->Set_coil_flag(is_use_coil);
	cout << "new class Run_mcs->Set_coil_flag  " << endl;

	Run_mcs_simulation->Set_image_dim_no_coil(dim_x, dim_y, dim_z);
//  Run_mcs_simulation->Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z);
	Run_mcs_simulation->Set_image_voxel_size(voxel_size);

	Run_mcs_simulation->Set_num_runs(nr_of_runs);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Set_thread_block_size(block_size);
	cout << "new class Run_mcs->Set_num_runs  " << endl;

	Run_mcs_simulation->Run_mcs_process();

	cout << "new class Run_mcs->Run_mcs_process  " << endl;

	cout << "end of simulation" << endl;

	of.close();

	if (Run_mcs_simulation != NULL) {
		delete Run_mcs_simulation;
		Run_mcs_simulation = NULL;

	}

}

int main(int argc, char *argv[]) {

	//cubic phantom
//	string patient_name="cubic_phantom";
//	string inputFolder="/data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/";
//	string frame_time=" ";

//Mcs_simu_cubic_phantom(patient_name,frame_time, inputFolder,nr_of_runs);

//////sphere phantom
//	string patient_name="XB1BN305N-BI-01";
//	string inputFolder="/data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/XB1BN305N-BI/"+patient_name+"/";
//	string frame_time="0-1800";
//

////OFOV Ge phantom
	string patient_name = "XB1BN286N-BI-09";
	string inputFolder =
			"/data/PET/mr_pet_temp/Ma/software/data/random_OFOV/Random-OFOV-activity-C11-2016-09-08/use_dead_scat_correction/"
					+ patient_name + "/";
	string frame_time = "0-1222";

	int nr_of_runs=5000;
//	int nr_of_runs = 1;

     Mcs_simu(patient_name,frame_time, inputFolder,nr_of_runs);

	return 1;

}
