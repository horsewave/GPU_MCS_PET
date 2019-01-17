//Name: all the structures used in GGMCS:
//Editor:Bo Ma
//Date:2016.12.29

#ifndef STRUCTURE_H
#define STRUCTURE_H

static const float gram  = 6.241509647e21f;
static const float mgram = gram * 0.001f;
static const float cm3   = 1.0e+03f;
static const float mole  = 1.0f;
static const float Avogadro = 6.02214179e23f;


struct my_float3 {
    float x, y, z;
};



struct my_int3 {
    int x, y, z;
};



#ifndef GPUPHANTOM
#define GPUPHANTOM
struct GPUPhantom {
    unsigned short int *data;
    unsigned int mem_data;
    my_float3 size_in_mm;
    my_int3 size_in_vox;
    my_float3 voxel_size;
    int nb_voxel_volume;
    int nb_voxel_slice;
};
#endif


#ifndef GPUPHANTOMMATERIALS
#define GPUPHANTOMMATERIALS
// GPU Structure of Arrays for phantom materials
struct GPUPhantomMaterials {
    unsigned int nb_materials;              // n
    unsigned int nb_elements_total;         // k

    unsigned short int *nb_elements;        // n
    unsigned short int *index;              // n
    unsigned short int *mixture;            // k
    float *atom_num_dens;                   // k
};
#endif

#ifndef GPUPHANTOMACTIVITIES
#define GPUPHANTOMACTIVITIES
// GPU SoA for phantom activities
struct GPUPhantomActivities{
    unsigned int nb_activities;
    float tot_activity;
    unsigned int *act_index;
    float *act_cdf;
    my_float3 size_in_mm;
 	my_int3 size_in_vox;
    my_float3 voxel_size;

};
#endif




#ifndef GPUSCANNER
#define GPUSCANNER
// GPU SoA for scanner geometry
struct GPUScanner{
    float cyl_radius;//187.96
    float cyl_halfheight;//96.25f
    float block_pitch;//32.5f
    float cta_pitch;//crystal size:2.5f
    float cax_pitch;//crystal size:2.5f
    unsigned short int ncass;//32
    unsigned short int nblock;//6 per cassette
    unsigned short int ncry_ta;//12 per block
    unsigned short int ncry_ax;//12 per block
    unsigned short int blocksize;//ncry_ta*ncry_ax=12*12;
    my_float3 halfsize;// real blocksize only with crystal:x:12*2.5=30;y:20mm;Z:12*2.5
    my_float3* pos;//position for each block in mm. matrix size:32*6=192
    my_float3* v0;// I don't know yet' matrix size:32=the cassette
    my_float3* v1;//same to v0
    my_float3* v2;//same to v0

    GPUPhantomMaterials mat; //phantom materials
};
#endif







typedef struct {
    double time;
    int id;
    int eventID;
    short int nCompton;
}Single ;


typedef struct {
    Single one;
    Single two;
    char type;
}Coincidence;


#ifndef GPUPHANTOMMATERIALS
#define GPUPHANTOMMATERIALS
// GPU Structure of Arrays for phantom materials
struct GPUPhantomMaterials {
    unsigned int nb_materials;//how many materials in the phantom.
    unsigned int nb_elements_total;         // k

    unsigned short int *nb_elements;        // n
    unsigned short int *index;              // n
    unsigned short int *mixture;            // k
    float *atom_num_dens;                   // k
};
#endif



#ifndef CPUSTACKPARTICLE
#define CPUSTACKPARTICLE
// Stack of particles, format data is defined as SoA
struct GPUParticleStack{
	float* E;
	float* dx;
	float* dy;
	float* dz;
	float* px;
	float* py;
	float* pz;
    float* tof;
	unsigned int* seed;
	unsigned char* endsimu;
  	unsigned char* active;
	unsigned long* table_x_brent;
    short int* crystalID;
    short int* nCompton;//Compton times in the phantom
    short int* nCompton_crystal;//compton times in crystal
    short int* nPhotoelectric_crystal;//photoelectric in crystal
    float* Edeposit;
	unsigned int size;
}; //
#endif

#endif




