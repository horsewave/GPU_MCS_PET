/*******************************
 *Name: run_mcs
 *Function: Run Monte Carlo simulation with multiple threads;
 *
 *Editor: Bo Ma
 *Time: 2016.10.20
 *Version:1.0
 *
 *Description:
 *        1: Interface function for the usr to call is :Run_mcs_process();
 *        2: Function to create multipul thread is :void Open_multi_threads();
 *        3:Function to run MCS is:static void* Mcs_simu(void *s_thread_arg);
 *        it must be static, and has the return type of "void*", and have the 
 *        parameter of "void*"
 *
 *        4:#ifndef __CINT__
            #include <pthread.h>
            #endif 
            This is necessary for the root to compile. or the rootcint will not recognize it.
 *
 *        5: Parameters have to be set before running is as follows:
 *
 *              Run_mcs *Run_mcs_simulation=new Run_mcs;

                Run_mcs_simulation->Set_file_name(fname_simu_run);
                Run_mcs_simulation->Set_material_path( path_element, path_material);
                Run_mcs_simulation->Set_phantom_path(path_rang);
                Run_mcs_simulation->Set_path_emi_img( emission_image_ori, emission_image_pro);
                Run_mcs_simulation->Set_path_att_img(atten_image_head_ori, atten_image_pro,atten_image_coils);
                Run_mcs_simulation->Set_stack_size(photon_num_per_run);
                Run_mcs_simulation->Set_GPU_device_num(nr_of_gpus);;
                Run_mcs_simulation-> Set_coil_flag( is_use_coil);
                Run_mcs_simulation->Set_num_runs(nr_of_runs);
   
                Run_mcs_simulation->Run_mcs_process();

*Questions need to solve:
                        1: how to new a class defined in a struct
                        2: pthread: if it is OK to pass a class to the parameter
                        3: how to copy a class
 * ****************************/


#ifndef RUN_MCS_H
#define RUN_MCS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


//#include "gpupetlib.h"
//#include "gpupetlib.h"
#include "GPUPETMonteCarlo.h"
#include "phanlib.h"
#include "plotlib.h"
#include "utilslib.h"
#include "scannerlib.h"
/*#include "../utils/phanlib.h"*/
//#include "../utils/plotlib.h"
//#include "../utils/utilslib.h"
/*#include "../utils/scannerlib.h"*/
//this is essential, or the cint can not pass
#ifndef __CINT__
#include <pthread.h>
#endif 
using namespace std;


// used to pass the scanner parametors to the Class of Scanner.
typedef struct{
    float cyl_radius;
    float cyl_halfheight;
    float block_pitch;
    float cta_pitch;
    float cax_pitch;
    unsigned short int ncass;
    unsigned short int nblock;
    unsigned short int ncry_ta;
    unsigned short int ncry_ax;
    float sizex;
    float sizey;
    float sizez;
    string cry_material;
    float window;
    float lower_threshold;

}s_my_Scanner;


//Used to pass the user's parametors to each thread'
struct s_multi_threads_arg
{
  int  nthreads;      //---number of threads
  int  thread_num;    //---id of thread 
  int  thread_status;  //---0: idle;  1: running; 2: finished;3:abort
  int  device_number;  // select the GPU device
  int nr_of_runs;// number of  run circles 
  GPUScanner s_gpu_scanner;      // scanner passed to GPU
  GPUPhantomActivities  s_gpu_activities;//activity passed to GPU
  GPUPhantom s_gpu_phantom;      //phantom passed to GPU
  GPUPhantomMaterials  s_gpu_materials;//materials passed to GPU
  s_my_Scanner s_scanner_para;// scanner parametor to create the CPU Scanner

  int stack_size;//photon pair numbers each run circle
  int gpu_block_size;//threads per block
  float gpu_time;//starting time of the GPU
  int n_lor;// size of the lor flie
  string file_name;// Input file name to save the simulation results.
  string file_name_log;// save the running log 
};




class Run_mcs
{
  public:

    //reconstructor
    Run_mcs();


    //Set the path of the material and elements file
    void Set_material_path(string path_element, string path_material);

    //Create an class of material for CPU and Load material and elements file
    void Load_material();

    //set the value range for each material of the phantom
    void Set_phantom_path(string path_rang);

    //create a phantom for cpu and load the emission image and the attenuation image
    void Load_phantom();

    //Set the threshold for the activity: if the activity value is less than the threshold, then set 0.
    void Set_phantom_activity_threshold(float threshold );


    //set the verbose 
    void Set_verbose(bool verbose);


    //Get the original emission image and set the path for the processed emission image
    void Set_path_emi_img(string path_emi_img_ori, string path_emi_img_pro);

    //same to Set_path_emi_img()
    void Set_path_att_img(string path_att_img_ori, string path_att_img_pro,string path_att_img_coil);
    
    //process the original emission image and attenuation to get the processed image for simulation
    void Read_image();
//    void Read_image(unsigned short int data_format);
    void Set_image_dim_no_coil(int dim_x, int dim_y, int dim_z);
    void Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z);
    void Set_image_voxel_size(float voxel_size);
    void Set_image_data_format(int size);

    //if or not using coil
    void Set_coil_flag(bool is_use_coil);

    //Set the scanner parameter for PET
    void Set_scanner_para(s_my_Scanner scanner_para);

    //Create a scanner using the scanner parameter
    void Load_scanner();

    //void Load_scanner(Scanner *cpu_scanner);
    
    // Multiple threads entry function to run MCS simulation:
    // It must be static function, and must be have a return type of void*, and must 
    // have a void * parameter to pass the user's data.
    static void* Mcs_simu(void *s_thread_arg);

    // used for tracking called by Mcs_simu(), Must be static
    static void Track_loop(GPUPETMonteCarlo *gpu_mcs_engine);

    //Set the GPU parameters, called by  Mcs_simu(),must be static, must set the device number first,
    //or the setting will fail.
    static void Set_GPU_engine(GPUPETMonteCarlo *mcs_engine, s_multi_threads_arg *s_thread_arg);

    //number of the simulating photons per run circle
    void Set_stack_size(int photon_number);

    //set the running number of the MCS
    void Set_num_runs(int num_runs);

    //set the thread number per block
       void Set_thread_block_size(int block_size);

    //set the device of the GPU, must put in the first line of the setting
    void Set_GPU_device_num(int device_number);

    //file name to save the simulation results.
    void Set_file_name(string fname);
    //void Set_lor_path(string lor_file_path);
    //
    //Write the simulation results to the defined file path
    static void Write_lor_file( float *lorval_true,float *lorval_scat,string file_name,int data_length);

    //Main interface function for the user  to call to run MCS
    void Run_mcs_process();

    //function to create multipul threads.
    void Open_multi_threads();


    //set the thread status: 0: idle;  1: running; 2: finished;3:abort
    void Set_thread_status(int thread_status);
    void Write_basic_info_log();
    ~Run_mcs();


  private:

    //enum for three dimention 
    enum EnumDim{DIM_X, DIM_Y, DIM_Z,DIM_NUMS };
    enum Enum_thread_Status{IDLE,RUNNING,FINISHED,ABORT};
    MaterialDataBase *m_db;
    Phantom *m_myphan;
    Scanner *m_myscanner;
    s_my_Scanner m_scanner_para;//parameters used to create Scanner for the cpu

    int m_image_data_format;
    

    
    bool m_is_use_coil;//flag if use coil
    int m_img_dim_no_coil[DIM_NUMS];//image dimentions 
    int m_img_dim_with_coil[DIM_NUMS];//image size if using coil
    float m_space;//voxel size for the image,default value:1.25
    int m_nlor;//file size for the lor data,default value:304*864*864;
    int m_stack_size;// photon pairs per run, defaut value:6000000
    int m_gpu_num;//total number of the gpu devices
    int m_nr_of_runs;// run numbers 
    string m_fname;//file name to save the simulation results
    string m_fname_log;//file name to save the running log
    string m_fpath_element;//path for elemental look up table for MCS
    string m_fpath_material;//path for  material look up table for MCS
    string m_fpath_range;//path for  the phantom material look up table for MCS
    float m_activity_threshold;// the threshold for the activity image,voxials which is below the value will set 0.
    string m_fpath_emission_img_original; //file path for the original emission image
    string m_fpath_emission_img_processed;// file path for the processed emission image, which is used directly for MCS
    string m_fpath_att_img_original;// file path for the original attenuation image, only head.
    string m_fpath_att_img_processed;//file path for the processed attenuation image
    string m_fpath_att_img_coil;//file path for the coil attenuation image.
    
    int m_gpu_block_size;// thread number per block.
    float m_gpu_time;// starting time of the GPU, as a standard for all the photons.
    bool m_verbose;

   int m_thread_status;  //---0: idle;  1: running; 2: finished;3:abort
   //static int m_thread_status;  //---0: idle;  1: running; 2: finished;3:abort

};


#endif


