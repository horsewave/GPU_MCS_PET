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

#include "../inc/run_mcs.h"

Run_mcs::Run_mcs()
  :m_db(NULL),
  m_myphan(NULL),
  m_myscanner(NULL),
 
  m_is_use_coil(true),
  m_space(1.25),
  m_image_data_format(4),
  m_stack_size(6000000),
  m_gpu_num(2),
  m_nr_of_runs(0),
  m_activity_threshold(0.0),
  m_gpu_block_size(320),
  m_gpu_time(0.0),
  m_verbose(false),
  m_thread_status(0)
 {
  m_fpath_element="../data/elts.dat";

   m_fpath_material="../data/mats.dat";
   m_fpath_range="../pet_data/range_brain.dat";
   m_nlor=304*864*864;

   m_img_dim_no_coil[DIM_X]=256;
  m_img_dim_no_coil[DIM_Y]=256;
  m_img_dim_no_coil[DIM_Z]=153;
  m_img_dim_with_coil[DIM_X]=320;
  m_img_dim_with_coil[DIM_Y]=320;
  m_img_dim_with_coil[DIM_Z]=153;



   m_scanner_para.cyl_radius=187.96;
   m_scanner_para.cyl_halfheight =96.25f;
   m_scanner_para.block_pitch =32.5f;
   m_scanner_para.cta_pitch =2.5f;
   m_scanner_para.cax_pitch =2.5f;
   m_scanner_para.ncass =32;
   m_scanner_para.nblock =6;
   m_scanner_para.ncry_ta =12 ;
   m_scanner_para.ncry_ax =12;
   m_scanner_para.sizex =30.0f;
   m_scanner_para.sizey =20.0f;
   m_scanner_para.sizez =30.0f;
   m_scanner_para.cry_material ="LSO";
   m_scanner_para.window =12.f;
   m_scanner_para.lower_threshold =0.42f;


}

////void Run_mcs::Show(Phantom num)

void Run_mcs::Run_mcs_process()
{

  //write the basic information
  Write_basic_info_log();
  ofstream of;
  of.open(m_fname_log.c_str(),std::ofstream::out | std::ofstream::app);
  double t2,time1;
  time1= get_time();
  t2=get_time();


  //start reading image
  of<<"Start to read original att and emission image: Read_image() "<<endl;

  Read_image();

  of<<"End of reading Image,time: "<<(get_time()-time1)<<" s"<<endl;
  printf("Read original att and emission image in %f s\n", get_time()-time1);

  //**** Read materials database
  of<<"Start to load material and element look up table: "<<endl;
  time1= get_time();
  Load_material();
  of<<"End of loading material and element look up table,time: "<< (get_time()-time1)<<endl;
  printf("Read materials database in %f s\n", get_time()-time1);

  // Read phantom
  of<<"Start to load phantom look up table: "<<endl;
  time1= get_time();
  Load_phantom();
  of<<"End of loading phantom look up table,time: "<< (get_time()-time1)<<endl;

  printf("Read phantom in %f s\n", get_time()-time1);

  // Create scanner
  of<<"Start to Create scanner for cpu: "<<endl;
  time1= get_time();

  Load_scanner();
  of<<"End of Create scanner for cpu,time: "<< (get_time()-time1)<<endl;
  printf("Create scanner and prepare for GPU in %f s\n", get_time()-time1);

  
  of<<"Start to create multiple threads and run MCS: Open_multi_threads()"<<endl;
 
  Open_multi_threads();
  
  of<<"********************************************************************"<<endl;
  of<<"All the threads are finished!"<<endl;

  
  char time_buf[80];
  //get the system time
  Get_sys_time(time_buf,80);

  of<<"Finish Time:"<<time_buf<<endl;

  of<<"********************************************************************"<<endl;
  of<<"Other GPU simulation details for each thread, please check the running log respectively!"<<endl;

  of.close();

}

void Run_mcs::Open_multi_threads()
{



 //---number of threads
  int nthreads=m_gpu_num;
  cout<<"thread number: "<<nthreads<<endl;
  double time1=get_time();
 
  //---thread records
  pthread_t threads[nthreads];//thread ID

  s_multi_threads_arg s_threads_arg[nthreads];// parameters used in each thread, only one, so usually it is a structure, including all the parameters; 

  //generating the user data for each thread
  for (int i=0; i<nthreads; i++)
  {
    std::stringstream ss;
    ss << i;
    s_threads_arg[i].file_name=m_fname+"_" + ss.str();
    s_threads_arg[i].file_name_log=s_threads_arg[i].file_name+"_log" ;
    s_threads_arg[i].nthreads=nthreads;
    s_threads_arg[i].thread_status=RUNNING;
    s_threads_arg[i].thread_num=i;
    s_threads_arg[i].device_number=i;
    s_threads_arg[i].nr_of_runs=m_nr_of_runs;
    s_threads_arg[i].stack_size=m_stack_size;
    s_threads_arg[i].gpu_block_size=m_gpu_block_size;
    s_threads_arg[i].gpu_time=m_gpu_time;
    s_threads_arg[i].n_lor=m_nlor;

    s_threads_arg[i].s_scanner_para=m_scanner_para;
    s_threads_arg[i].s_gpu_scanner=m_myscanner->get_scanner_for_GPU(*m_db);
    s_threads_arg[i].s_gpu_phantom=m_myphan->get_phantom_for_GPU();
    s_threads_arg[i].s_gpu_activities=m_myphan->get_activities_for_GPU();
    s_threads_arg[i].s_gpu_materials=m_myphan->get_materials_for_GPU(*m_db);
    

  }

  //write logs
  for (int i=0; i<nthreads; i++)
  {
    ofstream of;
    of.open(s_threads_arg[i].file_name_log.c_str(),std::ofstream::out | std::ofstream::app);


    char time_buf[80];
    
    //get the system time
    Get_sys_time(time_buf,80);
    of<<"********************************************************************"<<endl;
    of<<"Start MC simulation for GPU: "<<s_threads_arg[i].device_number<<endl;
    of<<"Start time: "<<time_buf<<endl;
    of<<"********************************************************************"<<endl;

    of<<"thread number is: "<<s_threads_arg[i].thread_num <<endl;

    of<<"File name for thread number: "<<i<<" is: "<<s_threads_arg[i].file_name <<endl;
    of<<"stack_size for thread:  "<<s_threads_arg[i].thread_num<<" is: "<< s_threads_arg[i].stack_size<<endl;
    of<<"Running circles for thread:  "<<s_threads_arg[i].thread_num<<" is: "<< s_threads_arg[i].nr_of_runs<<endl;
    of<<"thread number for each block of GPU in thread:  "<<s_threads_arg[i].thread_num<<"is: "<<s_threads_arg[i].gpu_block_size <<endl;
    of<<"m_nlor for thread: "<<s_threads_arg[i].thread_num<<" is: "<<s_threads_arg[i].n_lor <<endl;
    of<<"prepare scanner for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    of<<"prepare phantom for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    of<<"prepare material for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    of<<"thread status: "<<s_threads_arg[i].thread_status<<endl;

    of.close();


    if(m_verbose)
    {

    cout<<"********************************************************************"<<endl;
    cout<<"Start MC simulation for GPU: "<<s_threads_arg[i].device_number<<endl;
    cout<<"Start time: "<<time_buf<<endl;
    cout<<"********************************************************************"<<endl;
    cout<<"thread number is: "<<s_threads_arg[i].thread_num <<endl;
    cout<<"File name for thread number: "<<i<<"is: "<<s_threads_arg[i].file_name <<endl;
    cout<<"stack_size for thread:  "<<s_threads_arg[i].thread_num<<"is: "<< s_threads_arg[i].stack_size<<endl;
    cout<<"Running circles for thread:  "<<s_threads_arg[i].thread_num<<"is: "<< s_threads_arg[i].nr_of_runs<<endl;
    cout<<"thread number for each block of GPU in thread:  "<<s_threads_arg[i].thread_num<<"is: "<<s_threads_arg[i].gpu_block_size <<endl;
    cout<<"m_nlor for thread: "<<s_threads_arg[i].thread_num<<"is: "<<s_threads_arg[i].n_lor <<endl;
    cout<<"prepare scanner for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    cout<<"prepare phantom for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    cout<<"prepare material for GPU for thread: "<<s_threads_arg[i].thread_num<<endl;
    cout<<"thread status: "<<s_threads_arg[i].thread_status<<endl;


    }

  }

   ofstream of;
   of.open(m_fname_log.c_str(),std::ofstream::out | std::ofstream::app);

  for (int i=0; i<nthreads; i++)
  {

    of<< "start thread: " << i << endl;
    of<< "pthread_create(&threads[i], NULL, Mcs_simu, (void*)&s_threads_arg[i]); " << endl;
    cout << "start thread:" << i << endl;
    //---threaded execution 
    pthread_create(&threads[i], NULL, Mcs_simu, (void*)&s_threads_arg[i]);
  }

  
  //---wait_for_completion of all threads
  of<< "Info: wait for threads completion" << endl;
  of<< "pthread_join(threads[i], NULL)" << endl;
  cout << "Info: wait for thread completion" << endl;

  for (int i=0; i<nthreads; i++)
  {
    pthread_join(threads[i], NULL);//synchronize the threads;
  }
  of<< "Info: all threads completed,exit the function of: Open_multi_threads()" << endl;  
  cout << "Info: all threads completed" << endl;  
  of.close();
}


void Run_mcs::Set_material_path(string path_element, string path_material)
{
  m_fpath_element=path_element;
  m_fpath_material=path_material;
}
void Run_mcs:: Load_material()
{
  if(m_db!=NULL)
  {
    delete m_db;
    m_db=NULL;
  }
  m_db=new MaterialDataBase;
  m_db->load_elements(m_fpath_element.c_str());
  m_db->load_materials(m_fpath_material.c_str());

}
void Run_mcs::Set_phantom_path(string path_rang)
{
  m_fpath_range=path_rang;


}
void Run_mcs::Load_phantom()
{
  if(m_myphan!=NULL)
  {
    delete m_myphan;
  m_myphan=NULL;
  }
   m_myphan =new Phantom ;
   //load image
	if(m_is_use_coil)
  {
    m_myphan->load_from_raw(m_fpath_emission_img_processed.c_str(),m_img_dim_no_coil[DIM_X], m_img_dim_no_coil[DIM_Y], m_img_dim_no_coil[DIM_Z], m_fpath_att_img_processed.c_str() ,m_img_dim_with_coil[DIM_X],m_img_dim_with_coil[DIM_Y],m_img_dim_with_coil[DIM_Z], m_space, m_space,m_space );

  }
  else
  {
    m_myphan->load_from_raw(m_fpath_emission_img_processed.c_str(),m_img_dim_no_coil[DIM_X], m_img_dim_no_coil[DIM_Y], m_img_dim_no_coil[DIM_Z],  m_fpath_att_img_processed.c_str(),m_img_dim_no_coil[DIM_X] , m_img_dim_no_coil[DIM_Y],m_img_dim_no_coil[DIM_Z] ,m_space ,m_space ,m_space );

  }

    //load the attenuation look up table and generate an image of material 
    m_myphan->label_from_range(m_fpath_range.c_str());

    // Activity phantom according to image values, values under threshold are not used. usually set 0.
    m_myphan->activity_from_image(m_activity_threshold);

}
void Run_mcs:: Set_phantom_activity_threshold(float threshold )
{
  m_activity_threshold=threshold;

}

void Run_mcs::Set_path_emi_img(string path_emi_img_ori, string path_emi_img_pro)
{
  m_fpath_emission_img_original=path_emi_img_ori;
  m_fpath_emission_img_processed=path_emi_img_pro;


}
void Run_mcs::Set_path_att_img(string path_att_img_ori, string path_att_img_pro,string path_att_img_coil)
{
  m_fpath_att_img_original=path_att_img_ori;
  m_fpath_att_img_processed=path_att_img_pro;
  m_fpath_att_img_coil=path_att_img_coil;

}

void  Run_mcs::Set_verbose(bool verbose)
{
  m_verbose=verbose;
}



void  Run_mcs::Set_image_dim_no_coil(int dim_x, int dim_y, int dim_z)
{
  m_img_dim_no_coil[DIM_X]=dim_x;
  m_img_dim_no_coil[DIM_Y]=dim_y;
  m_img_dim_no_coil[DIM_Z]=dim_z;
  


}

void  Run_mcs::Set_image_dim_with_coil(int dim_x, int dim_y, int dim_z)
{
  m_img_dim_with_coil[DIM_X]=dim_x;
  m_img_dim_with_coil[DIM_Y]=dim_y;
  m_img_dim_with_coil[DIM_Z]=dim_z;


}

void  Run_mcs::Set_image_voxel_size(float voxel_size)
{
  m_space=voxel_size;

}


void  Run_mcs::Set_image_data_format(int size)
{
	m_image_data_format=size;
}


//Note: the input emission and attenuation image should be float
void  Run_mcs::Read_image()
{
  //
  int dimx=m_img_dim_no_coil[DIM_X];
  int dimy=m_img_dim_no_coil[DIM_Y];
  int dimz=m_img_dim_no_coil[DIM_Z];
  

  string na_image=m_fpath_emission_img_original;
  string atten_image_head=m_fpath_att_img_original;
  string atten_image_coils=m_fpath_att_img_coil;

  string emission_image=m_fpath_emission_img_processed;
  string atten_image=m_fpath_att_img_processed;
  bool usecoils=m_is_use_coil;

  int wordlength=sizeof(float);

    cout<<"dim x y z is "<<dimx<<"  "<<dimy<<"  "<<dimz<<endl;
    cout<<"worldlength is "<<wordlength<<endl;

  int nVoxels = dimx*dimy*dimz;
  double sum=0.0;	

  float *emission_data=new float[nVoxels];
  memset(emission_data, 0, sizeof(float)*nVoxels);

  float *atten_data=new float[nVoxels];
  memset(atten_data, 0, sizeof(float)*nVoxels);


   //read original image to memory
  ifstream fin;
  fin.open(na_image.c_str());

  if(fin.good()){
      cout<<"Reading emission file from "<<na_image.c_str()<<endl;
      fin.read((char *)emission_data, (dimx * dimy * dimz * wordlength));
      fin.close();
  }
  else
  {
    cout<<"Error opening emission file "<<na_image.c_str()<<endl;	
    fin.close();


  }
    
  fin.open(atten_image_head.c_str());		
  if(fin.good()){
    cout<<"Reading atten file from "<<atten_image_head.c_str()<<endl;
    fin.read((char *)atten_data, (dimx * dimy * dimz * wordlength));
    fin.close();
  }
  else
  {
    cout<<"Error opening atten file "<<atten_image_head.c_str()<<endl;	
    fin.close();

  }
    

  for (int i=0; i < nVoxels; i++)
  {
    sum+=emission_data[i];

  }
    


  for(int x = 0; x < dimx; x++){
    for(int y = 0; y < dimy; y++){
      for(int z = 0; z < dimz; z++){
        if(atten_data[z*dimx*dimy+y*dimx+x] > 0.02)
          //why?to reduce the time?
        	//continue;
          emission_data[z*dimx*dimy + y*dimx + x]/=(10*sum/nVoxels);
        else
          emission_data[z*dimx*dimy + y*dimx + x]=0.0;	
      }
    }
  }

    //write imge to disk
  ofstream fout;
  fout.open(emission_image.c_str());
  fout.write((char*) emission_data, (dimx * dimy * dimz * wordlength));
  fout.close();

  //if use coil
  if(usecoils){
    int dimx_coil=m_img_dim_with_coil[DIM_X];
    int dimy_coil=m_img_dim_with_coil[DIM_Y];
    int dimz_coil=m_img_dim_with_coil[DIM_Z];

    int nVoxels = dimx_coil*dimy_coil*dimz_coil;
    float *coil_data=new float[nVoxels];
    memset(coil_data, 0, sizeof(float)*nVoxels);

    fin.open(atten_image_coils.c_str());		
    if(fin.good()){
      cout<<"Reading coil file from "<<atten_image_coils.c_str()<<endl;
      fin.read((char *)coil_data, (dimx_coil * dimy_coil * dimz_coil * wordlength));
      fin.close();
    }
    else
      cout<<"Error opening coil file "<<atten_image_coils.c_str()<<endl;

    //add the head data to the coil data.
    for(int x = (dimx_coil - dimx)/2; x < (dimx_coil + dimx)/2; x++){
      for(int y = (dimy_coil - dimy)/2; y < (dimy_coil + dimy)/2; y++){
        for(int z = (dimz_coil - dimz)/2; z < (dimz_coil + dimz)/2; z++){
          if(coil_data[z*dimx_coil*dimy_coil + y*dimx_coil + x]==0)
            coil_data[z*dimx_coil*dimy_coil + y*dimx_coil + x]
              = atten_data[(z-(dimz_coil-dimz)/2)*dimx*dimy 
              + (y-(dimy_coil-dimy)/2)*dimx + x-(dimx_coil-dimx)/2];			
        }
      }
    }

    fout.open(atten_image.c_str());
    fout.write((char*) coil_data, (dimx_coil * dimy_coil * dimz_coil * wordlength));
    fout.close();
   delete [] coil_data;
   coil_data=NULL;

  }
  //if not use coil
  else{
    fout.open(atten_image.c_str());
    fout.write((char*) atten_data, (dimx * dimy * dimz * wordlength));
    fout.close();
  }

  //of.close();

  //delete occupied stack
  delete [] emission_data;
  emission_data=NULL;

  delete [] atten_data;
  atten_data=NULL;

 
}



void Run_mcs::Set_coil_flag(bool is_use_coil)
{
   m_is_use_coil=is_use_coil;

}
void Run_mcs::Set_scanner_para(s_my_Scanner scanner_para)
{
  m_scanner_para=scanner_para;

}
//void Run_mcs::Load_scanner(Scanner *cpu_scanner)
void Run_mcs::Load_scanner()
{

  if(m_myscanner!=NULL)
  {
    delete m_myscanner;
    m_myscanner=NULL;
  }
  m_myscanner=new Scanner(m_scanner_para.cyl_radius, m_scanner_para.cyl_halfheight, m_scanner_para.block_pitch, m_scanner_para.cta_pitch, m_scanner_para.cax_pitch, m_scanner_para.ncass, m_scanner_para.nblock, m_scanner_para.ncry_ta ,m_scanner_para.ncry_ax, m_scanner_para.sizex, m_scanner_para.sizey, m_scanner_para.sizez, m_scanner_para.cry_material, m_scanner_para.window, m_scanner_para.lower_threshold);

}

//set the thread number per block
void Run_mcs::Set_thread_block_size(int block_size)
{
	m_gpu_block_size=block_size;

}

void Run_mcs::Set_GPU_engine(GPUPETMonteCarlo *mcs_engine, s_multi_threads_arg *s_thread_arg)
{  

  // mcs_engine->set_device(s_thread_arg->device_number);must in the first line
  mcs_engine->set_device(s_thread_arg->device_number);

  mcs_engine->copy_materials_to_device(s_thread_arg->s_gpu_materials);
  mcs_engine->copy_phantom_to_device(s_thread_arg->s_gpu_phantom);
  mcs_engine->copy_activities_to_device(s_thread_arg->s_gpu_activities);
  mcs_engine->copy_scanner_to_device(s_thread_arg->s_gpu_scanner);
   

  mcs_engine->init_particle_stack(s_thread_arg->stack_size); // stack size
  mcs_engine->set_nb_of_particles(2*(s_thread_arg->stack_size)); // number of photons required
  mcs_engine->set_grid_device(s_thread_arg->gpu_block_size);      // 512 threads per block
  mcs_engine->set_time(s_thread_arg->gpu_time);
 

}

void* Run_mcs::Mcs_simu(void* s_thread_arg)
{
  s_multi_threads_arg *s_thread_arg_gpu=(s_multi_threads_arg *)s_thread_arg;

  ofstream of;
  of.open(s_thread_arg_gpu->file_name_log.c_str(),std::ofstream::out | std::ofstream::app);

  s_my_Scanner scan_para=s_thread_arg_gpu->s_scanner_para;

  of<<"start create cpu scanner class for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  //cout<<"start create cpu scanner class for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  Scanner cpu_scanner_in_thread(scan_para.cyl_radius, scan_para.cyl_halfheight, scan_para.block_pitch, scan_para.cta_pitch, scan_para.cax_pitch,scan_para.ncass, scan_para.nblock, scan_para.ncry_ta ,scan_para.ncry_ax, scan_para.sizex, scan_para.sizey, scan_para.sizez, scan_para.cry_material, scan_para.window, scan_para.lower_threshold);
  of<<"finish creating cpu scanner class for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  //cout<<" finish create scanner class for thread: "<< s_thread_arg_gpu->thread_num<<endl;


  double time1=get_time();
  double time2=get_time();

  of<<" start to create a mcs engine for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  //cout<<" start to create a mcs engine for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  GPUPETMonteCarlo *mcs_engine=new GPUPETMonteCarlo;

  of<<"set gpu engine  for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  //cout<<"set gpu engine  for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  Set_GPU_engine(mcs_engine,s_thread_arg_gpu);

  of<<"create lor data  for GPU :"<< s_thread_arg_gpu->device_number<<endl;
  //cout<<"create lor data for thread : "<< s_thread_arg_gpu->thread_num<<endl;

  float * lorval_true=new float[s_thread_arg_gpu->n_lor];
  memset(lorval_true,0,sizeof(float)*s_thread_arg_gpu->n_lor);	
  float * lorval_scat=new float[ s_thread_arg_gpu->n_lor];
  memset(lorval_scat,0,sizeof(float)* s_thread_arg_gpu->n_lor);	

  std::list<Single> SinglesList;

  int device_num_rand=s_thread_arg_gpu->device_number;
  srand(get_time()+device_num_rand*12345);

  string file_name_gpu=s_thread_arg_gpu->file_name;

  int run_num=s_thread_arg_gpu->nr_of_runs;

  of<<"*************************************************************"<<endl;
  of<<"Starting to MCS circle,  running number is: "<<run_num<<endl;
  of<<"*************************************************************"<<endl;
  //cout<<"Starting to MCS circle for GPU : "<< s_thread_arg_gpu->device_number<<", running number is: "<<run_num<<endl;


  for(int i=0; i<run_num;i++)
  {
    /*if(m_thread_status==ABORT)*/
    //{
      //break;
      //of<<"Manually aborted by the user!"<<endl;
      //cout<<"Manually aborted by the user!"<<endl;
    /*}*/
	double total_time=get_time();
//	  time1=get_time();
	  // To count simulated particles
    mcs_engine->init_particle_counter();

    time1=get_time();
    mcs_engine->init_PRNG(rand());             // Pseudo-Random Number Generator (seed)



    of<<" Inite GPU time : "<<(get_time()-time1)<<"s"<<endl;

    // Voxelized source
    time1 = get_time();
    mcs_engine->voxelized_source_b2b();

    of<<" voxelized_source_b2b time : "<<(get_time()-time1)<<"s"<<endl;



    // Tracking loop
    time1 = get_time();
    Track_loop(mcs_engine);
    of<<" Track_loop time : "<<(get_time()-time1)<<"s"<<endl;


    // Detection
    time1= get_time();
    mcs_engine->detection();
    of<<" Detection time : "<<(get_time()-time1)<<"s"<<endl;


    // DATA copy
        time1= get_time();
    GPUParticleStack g1 = mcs_engine->get_stack_gamma1();
    GPUParticleStack g2 = mcs_engine->get_stack_gamma2();
    of<<" data copy from GPU to CPU : "<<(get_time()-time1)<<" s"<<endl;

    total_time=get_time()-total_time;
    int one_million=1000000;

    int million_num= s_thread_arg_gpu->stack_size/one_million;
//    int million_num=one_million/s_thread_arg_gpu->stack_size;

    of<<" total time is:  "<< total_time<<" s"<<endl;
    total_time=total_time/million_num;
//    total_time=total_time*million_num;

    of<<" time for 1 millon 1 gpu is: "<< total_time<<" s"<<endl;


    time1=get_time();

    cpu_scanner_in_thread.process_to_singles(g1,g2, &SinglesList, s_thread_arg_gpu->s_gpu_activities.tot_activity);
//    cpu_scanner_in_thread.Get_hits_info(g1,g2);

    of<<" Total activity is : "<<s_thread_arg_gpu->s_gpu_activities.tot_activity<<"s"<<endl;
    of<<"process_to_singles time : "<<(get_time()-time1)<<"s"<<endl;

    time1 = get_time();
    bool first=0;
    if(i==0)
      first=true;

    bool last=0;
    if(i==run_num-1)
      last=true;

    cpu_scanner_in_thread.save_coincidences_lor(&SinglesList,file_name_gpu.c_str(),first,last,lorval_true,lorval_scat);
    of<<"save_coincidences_lor time : "<<(get_time()-time1)<<"s"<<endl;
    if(i<run_num-1)
      mcs_engine->free_counters();
  }

  of<<"******************************************** "<<endl;
  of<<"MC simulation finished! "<<endl;
  char time_buf[80];
  Get_sys_time(time_buf,80);
  of<<"Finishe time: "<<time_buf<<endl;
  double total_time=get_time()-time2;
  double ave_time_per_run=total_time/(s_thread_arg_gpu->nr_of_runs);
   of<<"The average time for each running is: "<<ave_time_per_run<<" s"<<endl;
  of<<"******************************************** "<<endl;

  of<<"Starting to save the results in the defined path! "<<time_buf<<endl;
  time1=get_time();

 Write_lor_file(lorval_true,lorval_scat,file_name_gpu,s_thread_arg_gpu->n_lor);

  of<<"Finished writing, time : "<<(get_time()-time1)<<"s"<<endl;

  if(lorval_true!=NULL)
  {
    delete[] lorval_true;
    lorval_true=NULL;

  }
  if(lorval_scat!=NULL)
  {
    delete[] lorval_scat;
    lorval_scat=NULL;

  }

  if(mcs_engine!=NULL)
  {
    delete  mcs_engine;
    mcs_engine=NULL;

  }
  cout<<"**************************************************"<<endl;
  cout<<"End of thread: "<<s_thread_arg_gpu->thread_num<<endl;
  of<<"End of thread: "<<s_thread_arg_gpu->thread_num<<endl;
  of<<"**************************************************"<<endl;
  Get_sys_time(time_buf,80);
  of<<"Finishe time: "<<time_buf<<endl;
  cout<<"Finishe time: "<<time_buf<<endl;
  cout<<"**************************************************"<<endl;

  //of.open
  of.close();
}

void Run_mcs::Track_loop(GPUPETMonteCarlo *gpu_mcs_engine )
{

  // Tracking loop
  double time1 = get_time();
  int step = 0;
  int nb_particles = gpu_mcs_engine->get_nb_of_particles();
 cout<<"photon numbers to track: "<< nb_particles<<endl;

  int nb_simulated = 0;
  while (nb_simulated < nb_particles) {
    ++step;

    // navigation
    gpu_mcs_engine->navigation();

    // update counter
    nb_simulated = gpu_mcs_engine->get_nb_of_simulated();

    // watchdog
    if (step > 1500) {

      printf("WARNING - Reach the maximum number of steps\n");
      break;
    }

  }
  printf("Navigation %f s    step %i - sim %i/%i\n", get_time()-time1, 
      step, nb_simulated, nb_particles);
}


void Run_mcs::Set_stack_size(int photon_number)
{
  m_stack_size=photon_number;

}
void Run_mcs::Set_num_runs(int num_runs)
{
  
  m_nr_of_runs=num_runs;
}
void Run_mcs::Set_GPU_device_num(int device_number)
{
  m_gpu_num=device_number;
  
}

void Run_mcs::Set_file_name(string fname)
{

  m_fname=fname;
  m_fname_log=fname+"_log";


}

void Run_mcs::Write_lor_file(float *lorval_true,float *lorval_scat,string file_name,int data_length )
{

  string fpath_lor_true=file_name+"_true.flor";
  string fpath_lor_scat=file_name+"_scatter.flor";
  FILE *f1=NULL;
  f1=fopen(fpath_lor_scat.c_str(),"wb");
  fwrite(lorval_scat,sizeof(float), data_length,f1);
  fclose(f1);

  f1=NULL;
  f1=fopen(fpath_lor_true.c_str(),"wb");
  fwrite(lorval_true,sizeof(float), data_length,f1);
  fclose(f1);

}

 void Run_mcs::Set_thread_status(int thread_status)
{
  m_thread_status=thread_status;

}
void  Run_mcs::Write_basic_info_log()
{
  ofstream of;
  of.open(m_fname_log.c_str(),std::ofstream::out | std::ofstream::app);
  of<<"Title: GPU based Monte Carlo Simulation"<<endl;
  of<<"Editor: Bo Ma"<<endl;
  char time_buf[80];
  Get_sys_time(time_buf,80);
  of<<"Simulation Time: "<<time_buf<<endl;

  of<<"Simulation basic Information: "<<endl;
  of<<"******************************************************************** "<<endl;

  of<<"Number of GPU: " <<m_gpu_num<<endl;
  of<<"Running circles for each GPU: " <<m_nr_of_runs<<endl;
  of<<"Photon pairs for each run circle: " <<m_stack_size<<endl;
  of<<"Activity threshold: "<<m_activity_threshold<<endl;
  of<<"Thread amount per block: "<<m_gpu_block_size<<endl;

  of<<"Path for original emission image: " <<m_fpath_emission_img_original<<endl;
  of<<"Path for original attenuation(only head) image: " <<m_fpath_att_img_original<<endl;
  of<<"If or not using coil(0: No; 1:Yes): "<<m_is_use_coil<<endl;
  of<<"Path for the coil image: "<<m_fpath_att_img_coil<<endl;
  of<<"Path for element looking up table: "<<m_fpath_element<<endl;
  of<<"Path for material looking up table: "<<m_fpath_material<<endl;
  of<<"Path for phantom range looking up table: "<<m_fpath_range<<endl;
  of<<"Path for the simulataion results: "<<m_fname<<endl;
  of<<"******************************************************************** "<<endl;
  of<<"******************************************************************** "<<endl;
  of<<"******************************************************************** "<<endl;
  of<<"******************************************************************** "<<endl;
  of.close();


}


Run_mcs::~Run_mcs()
{
 if(m_db!=NULL)
  {
    delete  m_db;
   m_db=NULL;

  }
if(m_myphan!=NULL)
  {
    delete  m_myphan;
    m_myphan=NULL;

  }
if(m_myscanner!=NULL)
  {
    delete  m_myscanner;
    m_myscanner=NULL;

  }



}

