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



///// Utils ///////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>

//#include "../inc/gpupetlib.h"
#include "../inc/pe_cs_const_table.h"
#include "../inc/gpupetkernel.cuh"

// Inverse vector
__device__ float3 vec3_inverse(float3 u) {
    return make_float3(__fdividef(1.0f, u.x), __fdividef(1.0f, u.y), __fdividef(1.0f, u.z));
}

// rotateUz, function from CLHEP
__device__ float3 rotateUz(float3 vector, float3 newUz) {
	float u1 = newUz.x;
	float u2 = newUz.y;
	float u3 = newUz.z;
	float up = u1*u1 + u2*u2;

	if (up>0) {
		up = sqrtf(up);
		float px = vector.x,  py = vector.y, pz = vector.z;
		vector.x = __fdividef(u1*u3*px - u2*py, up) + u1*pz;
		vector.y = __fdividef(u2*u3*px + u1*py, up) + u2*pz;
		vector.z =              -up*px +              u3*pz;
    }
	else if (u3 < 0.) { vector.x = -vector.x; vector.z = -vector.z; } // phi=0  theta=gpu_pi

	return make_float3(vector.x, vector.y, vector.z);
}

// Return the next voxel boundary distance, it is used by the standard navigator
#define INF 1.0e30f
#define EPS 1.0e-04f
__device__ float get_boundary_voxel_by_raycasting(int4 vox, float3 p, float3 d, float3 res) {
    
    
	float xmin, xmax, ymin, ymax, zmin, zmax;
    float3 di = vec3_inverse(d);
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	
    // Define the voxel bounding box
    xmin = (d.x > 0 && p.x > (vox.x+1) * res.x - EPS) ? (vox.x+1) * res.x : vox.x*res.x;
    ymin = (d.y > 0 && p.y > (vox.y+1) * res.y - EPS) ? (vox.y+1) * res.y : vox.y*res.y;
    zmin = (d.z > 0 && p.z > (vox.z+1) * res.z - EPS) ? (vox.z+1) * res.z : vox.z*res.z;
    
    xmax = (d.x < 0 && p.x < xmin + EPS) ? xmin-res.x : xmin+res.x;
    ymax = (d.y < 0 && p.y < ymin + EPS) ? ymin-res.y : ymin+res.y;
    zmax = (d.z < 0 && p.z < zmin + EPS) ? zmin-res.z : zmin+res.z;
    
    tmin = -INF;
    tmax = INF;
    
    // on x
    if (fabsf(d.x) > EPS) {
        tmin = (xmin - p.x) * di.x;
        tmax = (xmax - p.x) * di.x;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
    }
    // on y
    if (fabsf(d.y) > EPS) {
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
    }
    // on z
    if (fabsf(d.z) > EPS) {
        tzmin = (zmin - p.z) * di.z;
        tzmax = (zmax - p.z) * di.z;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
    }

    return tmax;
}
#undef INF
#undef EPS
#define INF 1e9f
#define EPSILON 1e-6f
// Ray/AABB - Smits algorithm
__device__ float mc_hit_AABB(float px, float py, float pz,     // Ray definition (pos, dir) 
                  float dx, float dy, float dz,
                  float xmin, float xmax,           // AABB definition
                  float ymin, float ymax, 
                  float zmin, float zmax) {
    
    
    float idx, idy, idz;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	
    tmin = 0.0f;
    tmax = INF;

    // on x
    if (fabs(dx) < EPSILON) {
        if (px < xmin || px > xmax) {return 0;}
    } else {
        idx = 1.0f / dx;
        tmin = (xmin - px) * idx;
        tmax = (xmax - px) * idx;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return 0;}
    }
    // on y
    if (fabs(dy) < EPSILON) {
        if (py < ymin || py > ymax) {return 0;}
    } else {
        idy = 1.0f / dy;
        tymin = (ymin - py) * idy;
        tymax = (ymax - py) * idy;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
        if (tmin > tmax) {return 0;}
    }
    // on z
    if (fabs(dz) < EPSILON) {
        if (pz < zmin || pz > zmax) {return 0;}
    } else {
        idz = 1.0f / dz;
        tzmin = (zmin - pz) * idz;
        tzmax = (zmax - pz) * idz;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
        if (tmin > tmax) {return 0;}
    }
    
    return tmin;
}

// Ray/OBB
__device__ float mc_hit_OBB(float px, float py, float pz, // Ray definition (pos, dir) 
                 float dx, float dy, float dz,
                 float xmin, float xmax,       // Slabs are defined in OBB's space
                 float ymin, float ymax, 
                 float zmin, float zmax,
                 float cx, float cy, float cz, // OBB center
                 float ux, float uy, float uz, // OBB orthognal space (u, v, w)
                 float vx, float vy, float vz,
                 float wx, float wy, float wz) {
    
    
    // Transform the ray in OBB's space, then do AABB
    float px_obb = px-cx;
    float py_obb = py-cy;
    float pz_obb = pz-cz;
    px = px_obb*ux + py_obb*uy + pz_obb*uz;
    py = px_obb*vx + py_obb*vy + pz_obb*vz;
    pz = px_obb*wx + py_obb*wy + pz_obb*wz;
    float ddx = dx*ux + dy*uy + dz*uz;
    float ddy = dx*vx + dy*vy + dz*vz;
    float ddz = dx*wx + dy*wy + dz*wz;

    return mc_hit_AABB(px, py, pz, ddx, ddy, ddz,
                       xmin, xmax, ymin, ymax, zmin, zmax);

}
#undef INF
#undef EPSILON

// Return the distance to the block boundary
#define INF 1.0e30f
#define EPS 1.0e-06f
__device__ float get_boundary_block_by_raycasting(float3 p, float3 d, float3 halfsize) {
    
    
	float xmin, xmax, ymin, ymax, zmin, zmax;
    float3 di = vec3_inverse(d);
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	
    // Define the bounding box
    xmin = -halfsize.x;
    ymin = -halfsize.y;
    zmin = -halfsize.z;
    xmax = +halfsize.x;
    ymax = +halfsize.y;
    zmax = +halfsize.z;
    
    tmin = -INF;
    tmax = INF;
    
    // on x
    if (fabsf(d.x) > EPS) {
        tmin = (xmin - p.x) * di.x;
        tmax = (xmax - p.x) * di.x;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
    }
    // on y
    if (fabsf(d.y) > EPS) {
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
    }
    // on z
    if (fabsf(d.z) > EPS) {
        tzmin = (zmin - p.z) * di.z;
        tzmax = (zmax - p.z) * di.z;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
    }

    return tmax;
}
#undef INF
#undef EPS

#define INF 1.0e9f
__device__ float Move_To_Next_Block(float3& position, float3& direction, int& currentid, GPUScanner scanner, float d_min, int id){
    
    int cassid=currentid/scanner.nblock; 
    int currblock=currentid%scanner.nblock;

    float3 direction2;
    float3 position2; 

    direction2.x=  direction.x *scanner.v0[cassid].x + direction.y *scanner.v1[cassid].x + direction.z *scanner.v2[cassid].x;
    direction2.y=  direction.x *scanner.v0[cassid].y + direction.y *scanner.v1[cassid].y + direction.z *scanner.v2[cassid].y;
    direction2.z=  direction.x *scanner.v0[cassid].z + direction.y *scanner.v1[cassid].z + direction.z *scanner.v2[cassid].z;

    position2.x=  position.x *scanner.v0[cassid].x + position.y *scanner.v1[cassid].x + position.z *scanner.v2[cassid].x + scanner.pos[currentid].x;
    position2.y=  position.x *scanner.v0[cassid].y + position.y *scanner.v1[cassid].y + position.z *scanner.v2[cassid].y + scanner.pos[currentid].y;
    position2.z=  position.x *scanner.v0[cassid].z + position.y *scanner.v1[cassid].z + position.z *scanner.v2[cassid].z + scanner.pos[currentid].z;

    const int c_min=max(cassid-1,0);
    const int c_max=min(cassid+1,scanner.ncass-1);
    const int b_min=max(currblock-1,0);
    const int b_max=min(currblock+1,scanner.nblock-1);

    float dmin=INF;
    int temp=-1;

    for(int icass=c_min; icass<=c_max;icass++){
        for(int ib=b_min; ib<=b_max;ib++){
            
            const int blockid=icass*scanner.nblock+ib;
            if(blockid!=currentid){
                float distance=0.0f;
                distance= mc_hit_OBB(position2.x, position2.y, position2.z,               
                    direction2.x, direction2.y, direction2.z,
                    -scanner.halfsize.x,+scanner.halfsize.x,-scanner.halfsize.y,+scanner.halfsize.y,-scanner.halfsize.z,+scanner.halfsize.z,
                    scanner.pos[blockid].x, scanner.pos[blockid].y, scanner.pos[blockid].z,
                    scanner.v0[icass].x, scanner.v0[icass].y, scanner.v0[icass].z,
                    scanner.v1[icass].x, scanner.v1[icass].y, scanner.v1[icass].z,
                    scanner.v2[icass].x, scanner.v2[icass].y, scanner.v2[icass].z);
          
                if(distance>0 && distance<dmin){
                    dmin=distance;    
                    temp=blockid;
                }
            }
        }
    }

    if(temp!=-1){

      cassid=temp/scanner.nblock; 
      position2.x -=  scanner.pos[temp].x;
      position2.y -=  scanner.pos[temp].y;
      position2.z -=  scanner.pos[temp].z;

      position.x=  position2.x*scanner.v0[cassid].x + position2.y*scanner.v0[cassid].y + position2.z*scanner.v0[cassid].z;
      position.y=  position2.x*scanner.v1[cassid].x + position2.y*scanner.v1[cassid].y + position2.z*scanner.v1[cassid].z;
      position.z=  position2.x*scanner.v2[cassid].x + position2.y*scanner.v2[cassid].y + position2.z*scanner.v2[cassid].z;

      direction.x=  direction2.x*scanner.v0[cassid].x + direction2.y*scanner.v0[cassid].y + direction2.z*scanner.v0[cassid].z;
      direction.y=  direction2.x*scanner.v1[cassid].x + direction2.y*scanner.v1[cassid].y + direction2.z*scanner.v1[cassid].z;
      direction.z=  direction2.x*scanner.v2[cassid].x + direction2.y*scanner.v2[cassid].y + direction2.z*scanner.v2[cassid].z;
    }
    currentid=temp;
    
    return dmin;
}
#undef INF

// Linear search
__device__ int linear_search(float *val, float key) {
    int pos=0; while(val[pos] < key) ++pos;
    return pos;
}

// Relative search
__device__ int relative_search(float *val, float key, int n) {
    int pos = (int)(key * (float)n);
    if (val[pos] <= key) {
        while(val[pos] < key) ++pos;
    } else {
        while(val[pos] >= key) --pos;
        ++pos; // correct undershoot
    }
    return pos;
}

// Binary search
__device__ int binary_search(float *val, float key, int n) {
    int min=0, max=n, mid;
    while (min < max) {
        mid = (min + max) >> 1;
        if (key > val[mid]) {
            min = mid + 1;
        } else {
            max = mid;
        }
    }
    return min; 
}


///// PRNG Brent xor256 //////////////////////////////////////////////////

// Brent PRNG integer version
//__device__ unsigned long weyl;
__device__ unsigned long brent_int(unsigned int index, unsigned long *device_x_brent, unsigned long seed)

{
	
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64) 
#define wlen (64*UINT64 +  32*UINT32)
#define r    (4*UINT64 + 8*UINT32)
#define s    (3*UINT64 +  3*UINT32)
#define a    (37*UINT64 +  18*UINT32)
#define b    (27*UINT64 +  13*UINT32)
#define c    (29*UINT64 +  14*UINT32)
#define d    (33*UINT64 +  15*UINT32)
#define ws   (27*UINT64 +  16*UINT32) 

	int z, z_w, z_i_brent;	
	if (r==4){
		z=6; z_w=4; z_i_brent=5;}
	else{
		z=10; z_w=8; z_i_brent=9;}
	
	unsigned long w = device_x_brent[z*index + z_w];
	unsigned long i_brent = device_x_brent[z*index + z_i_brent];
	unsigned long zero = 0;
	unsigned long t, v;
	int k;
	
	if (seed != zero) { // Initialisation necessary
		// weyl = odd approximation to 2**wlen*(3-sqrt(5))/2.
		if (UINT32) 
			weyl = 0x61c88647;
		else 
			weyl = ((((unsigned long)0x61c88646)<<16)<<16) + (unsigned long)0x80b583eb;
		
		v = (seed!=zero)? seed:~seed;  // v must be nonzero
		
		for (k = wlen; k > 0; k--) {   // Avoid correlations for close seeds
			v ^= v<<10; v ^= v>>15;    // Recurrence has period 2**wlen-1
			v ^= v<<4;  v ^= v>>13;    // for wlen = 32 or 64
		}
		for (w = v, k = 0; k < r; k++) { // Initialise circular array
			v ^= v<<10; v ^= v>>15; 
			v ^= v<<4;  v ^= v>>13;
			device_x_brent[k + z*index] = v + (w+=weyl);              
		}
		for (i_brent = r-1, k = 4*r; k > 0; k--) { // Discard first 4*r results
			t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index];   t ^= t<<a;  t ^= t>>b;			
			v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];	v ^= v<<c;  v ^= v>>d;       
			device_x_brent[i_brent + z*index] = t^v;  
		}
    }
    
	// Apart from initialisation (above), this is the generator
	t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index]; // Assumes that r is a power of two
	v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];       // Index is (i-s) mod r
	t ^= t<<a;  t ^= t>>b;                                       // (I + L^a)(I + R^b)
	v ^= v<<c;  v ^= v>>d;                                       // (I + L^c)(I + R^d)
	device_x_brent[i_brent + z*index] = (v ^= t); 				 // Update circular array                 
	w += weyl;                                                   // Update Weyl generator
	
	device_x_brent[z*index + z_w] = w;
	device_x_brent[z*index + z_i_brent] = i_brent;
	
	return (v + (w^(w>>ws)));  // Return combination
	
#undef UINT64
#undef UINT32
#undef wlen
#undef r
#undef s
#undef a
#undef b
#undef c
#undef d
#undef ws 
}	

// Brent PRNG real version
__device__ double Brent_real(int index, unsigned long *device_x_brent, unsigned long seed)

{
	
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64) 
#define UREAL64 (sizeof(double)>>3)
#define UREAL32 (1 - UREAL64)
	
	// sr = number of bits discarded = 11 for double, 40 or 8 for float
	
#define sr (11*UREAL64 +(40*UINT64 + 8*UINT32)*UREAL32)
	
	// ss (used for scaling) is 53 or 21 for double, 24 for float
	
#define ss ((53*UINT64 + 21*UINT32)*UREAL64 + 24*UREAL32)
	
	// SCALE is 0.5**ss, SC32 is 0.5**32
	
#define SCALE ((double)1/(double)((unsigned long)1<<ss)) 
#define SC32  ((double)1/((double)65536*(double)65536)) 
	
	double res;
	
	res = (double)0; 
	while (res == (double)0)  // Loop until nonzero result.
    {   // Usually only one iteration.
		res = (double)(brent_int(index, device_x_brent, seed)>>sr);     // Discard sr random bits.
		seed = (unsigned long)0;                                        // Zero seed for next time.
		if (UINT32 && UREAL64)                                          // Need another call to xor4096i.
			res += SC32*(double)brent_int(index, device_x_brent, seed); // Add low-order 32 bits.
    }
	return (SCALE*res); // Return result in (0.0, 1.0).
	
#undef UINT64
#undef UINT32
#undef UREAL64
#undef UREAL32
#undef SCALE
#undef SC32
#undef sr
#undef ss
}




// Init rand seed for GPU for each kernel. But curand_init()is too time_consuming.
__global__ void kernel_random_seed_init(GPUParticleStack stackpart,curandState *state) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id < stackpart.size) {

//		curandState state;
//		curand_init(clock64(), id, 0, &state);
		 curand_init(clock64(), id, 0, &state[id]);
//
//		 stackpart.seed[id]=curand(&state[id]);
		 stackpart.seed[id]=curand_uniform(&state[id])*RAND_MAX;

//	     printf("the id %d,random number is %d\n ",id,stackpart.seed[id]);

		}


}


// Init Brent seed
__global__ void kernel_brent_init(GPUParticleStack stackpart) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id < stackpart.size) {
		unsigned int seed = stackpart.seed[id];
		float dummy = brent_int(id, stackpart.table_x_brent, seed);
	}
}

///// Physics ///////////////////////////////////////////////////////////////

///// PhotoElectric /////

// PhotoElectric Cross Section Per Atom (Standard)
__device__ float PhotoElec_CSPA(float E, unsigned short int Z) {
	// from Sandia data, the same for all Z
	float Emin = fmax(PhotoElec_std_IonizationPotentials[Z]*1e-6f, 0.01e-3f);
	if (E < Emin) {return 0.0f;}
	
	int start = PhotoElec_std_CumulIntervals[Z-1];
	int stop = start + PhotoElec_std_NbIntervals[Z];
	int pos=stop;
	while (E < PhotoElec_std_SandiaTable[pos][0]*1.0e-3f){--pos;}
	float AoverAvo = 0.0103642688246f * __fdividef((float)Z, PhotoElec_std_ZtoAratio[Z]);
	float rE = __fdividef(1.0f, E);
	float rE2 = rE*rE;

	return rE * PhotoElec_std_SandiaTable[pos][1] * AoverAvo * 0.160217648e-22f
		+ rE2 * PhotoElec_std_SandiaTable[pos][2] * AoverAvo * 0.160217648e-25f
		+ rE * rE2 * PhotoElec_std_SandiaTable[pos][3] * AoverAvo * 0.160217648e-28f
		+ rE2 * rE2 * PhotoElec_std_SandiaTable[pos][4] * AoverAvo * 0.160217648e-31f;
}

// Compute the total Compton cross section for a given material
__device__ float PhotoElec_CS(GPUPhantomMaterials materials, 
                              unsigned short int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
	// Model standard
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
		CS += (materials.atom_num_dens[index+i] * 
               PhotoElec_CSPA(E, materials.mixture[index+i]));
	}
	return CS;
}

// Photoelectric effect without photoelectron
__device__ void PhotoElec_SampleSecondaries(GPUParticleStack photons, 
                                            unsigned int id, int *ct_d_sim) {
    // Absorbed the photon
    photons.endsimu[id] = 1; // stop the simulation
    photons.active[id] = 0; // declare the photon inactive
    atomicAdd(ct_d_sim, 1);  // count simulated primaries
}

///// Compton /////

// Compton Cross Section Per Atom (Standard - Klein-Nishina)
__device__ float Compton_CSPA(float E, unsigned short int Z) {
	float CrossSection = 0.0;
	if (Z<1 || E < 1e-4f) {return CrossSection;}
	float p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
	float p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
	float p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
	float p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
	float T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
	float d1, d2, d3, d4, d5;

	d1 = __fdividef(fmaxf(E, T0), 0.510998910f); // X
	CrossSection = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1);

	if (E < T0) {
		d1 = __fdividef(T0+1.0e-3f, 0.510998910f); // X
		d2 = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1); // sigma
		d3 = __fdividef(-T0 * (d2 - CrossSection), CrossSection*1.0e-3f); // c1
		d4 = (Z > 1.5f)? 0.375f-0.0556f*__logf(Z) : 0.15f; // c2
		d5 = __logf(__fdividef(E, T0)); // y
		CrossSection *= __expf(-d5 * (d3 + d4*d5));
	}
	
    return CrossSection;
}

// Compute the total Compton cross section for a given material
__device__ float Compton_CS(GPUPhantomMaterials materials, 
                                     unsigned short int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
	// Model standard
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
		CS += (materials.atom_num_dens[index+i] * 
               Compton_CSPA(E, materials.mixture[index+i]));
	}
	return CS;
}

// Compton Scatter without secondary electron (Standard - Klein-Nishina)
__device__ void Compton_SampleSecondaries(GPUParticleStack photons, 
                                           unsigned int id, int *d_ct_sim) {
	float gamE0 = photons.E[id];

	float E0 = __fdividef(gamE0, 0.510998910f);
    float3 gamDir0 = make_float3(photons.dx[id], photons.dy[id], photons.dz[id]);

    // sample the energy rate of the scattered gamma

	float epszero = __fdividef(1.0f, (1.0f + 2.0f * E0));
	float eps02 = epszero*epszero;
	float a1 = -__logf(epszero);
	float a2 = __fdividef(a1, (a1 + 0.5f*(1.0f-eps02)));

	float greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
	do {
		if (a2 > Brent_real(id, photons.table_x_brent, 0)) {
			eps = __expf(-a1 * Brent_real(id, photons.table_x_brent, 0));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * Brent_real(id, photons.table_x_brent, 0);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		sint2 = onecost * (2.0f - onecost);
		greject = 1.0f - eps * __fdividef(sint2, 1.0f + eps2);
	} while (greject < Brent_real(id, photons.table_x_brent, 0));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
	cosTheta = 1.0f - onecost;
	sinTheta = sqrt(sint2);
	phi = Brent_real(id, photons.table_x_brent, 0) * gpu_twopi;

    // update the scattered gamma

    float3 gamDir1 = make_float3(sinTheta*__cosf(phi), sinTheta*__sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    photons.dx[id] = gamDir1.x;
    photons.dy[id] = gamDir1.y;
    photons.dz[id] = gamDir1.z;
    photons.nCompton[id]++;
    float gamE1  = gamE0 * eps;
    if (gamE1 > 0.4f) {photons.E[id] = gamE1;}
//Bo
//    if (gamE1 > 1.0e-06f) {photons.E[id] = gamE1;}

    else {
        photons.endsimu[id] = 1; // stop this particle
        photons.active[id] = 0; // declare the photon inactive
        atomicAdd(d_ct_sim, 1);  // count simulated primaries
    }
}

// Compton Scatter without secondary electron (Standard - Klein-Nishina)
__device__ void Compton_SampleSecondaries_Detector(GPUParticleStack photons, 
                                           unsigned int id, float3& direction, 
                                           float &energy, bool &act) {
	float gamE0 = energy;

	float E0 = __fdividef(gamE0, 0.510998910f);
    float3 gamDir0 = make_float3(direction.x, direction.y, direction.z);

    // sample the energy rate of the scattered gamma

	float epszero = __fdividef(1.0f, (1.0f + 2.0f * E0));
	float eps02 = epszero*epszero;
	float a1 = -__logf(epszero);
	float a2 = __fdividef(a1, (a1 + 0.5f*(1.0f-eps02)));

	float greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
	do {
		if (a2 > Brent_real(id, photons.table_x_brent, 0)) {
			eps = __expf(-a1 * Brent_real(id, photons.table_x_brent, 0));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * Brent_real(id, photons.table_x_brent, 0);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		sint2 = onecost * (2.0f - onecost);
		greject = 1.0f - eps * __fdividef(sint2, 1.0f + eps2);
	} while (greject < Brent_real(id, photons.table_x_brent, 0));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
	cosTheta = 1.0f - onecost;
	sinTheta = sqrt(sint2);
	phi = Brent_real(id, photons.table_x_brent, 0) * gpu_twopi;

    // update the scattered gamma

    float3 gamDir1 = make_float3(sinTheta*__cosf(phi), sinTheta*__sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    direction.x = gamDir1.x;
    direction.y = gamDir1.y;
    direction.z = gamDir1.z;
    float gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {
    	energy = gamE1;
//    	printf("Compton:  input energy: %f Mev ,inactive energy is: %f Mev\n",gamE0,gamE1);

    }
    else {

    	printf("******Compton:  input energy: %f Mev ,inactive energy is: %f Mev*****\n",gamE0,gamE1);
        photons.active[id] = 0; // declare the photon inactive
        energy=0;
        act=false;
    }
}

///// Source ///////////////////////////////////////////////////////////////

// Voxelized source
__global__ void kernel_voxelized_source_b2b(GPUParticleStack d_g1, 
                                            GPUParticleStack d_g2,
                                            GPUPhantomActivities d_act,
                                            float3 phantom_size_in_mm) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (id >= d_g1.size) return;

    float jump = (float)(d_act.size_in_vox.x*d_act.size_in_vox.y);
    float ind, x, y, z;

    // use cdf to find the next emission spot 
    //rnd:[0,1] is the position in the acitvity density array.act_cdf
    float rnd = Brent_real(id, d_g1.table_x_brent, 0);
    //according rnd,find the most possible index in act_cdf; and then find the voxel index in the act_index array.
    int pos = binary_search(d_act.act_cdf, rnd, d_act.nb_activities);

    // convert position index to emitted position
    ind = (float)d_act.act_index[pos];
    z = floor(ind / jump);
    ind -= (z*jump);
    y = floor(ind / (float)d_act.size_in_vox.x);
    x = ind - y*d_act.size_in_vox.x;

    // random poisiton within the voxel

    x += Brent_real(id, d_g1.table_x_brent, 0);
    y += Brent_real(id, d_g1.table_x_brent, 0);
    z += Brent_real(id, d_g1.table_x_brent, 0);

    // convert in mm
    x *= d_act.voxel_size.x;
    y *= d_act.voxel_size.y;
    z *= d_act.voxel_size.z;

	// shift according to center of phantom
    x += phantom_size_in_mm.x/2 - d_act.size_in_mm.x/2;
    y += phantom_size_in_mm.y/2 - d_act.size_in_mm.y/2;
    z += phantom_size_in_mm.z/2 - d_act.size_in_mm.z/2;

    // random orientation
    float phi = Brent_real(id, d_g1.table_x_brent, 0);
    float theta =Brent_real(id, d_g1.table_x_brent, 0);
    phi *= gpu_twopi;
    theta = acosf(1.0f - 2.0f*theta);

    // compute direction vector
    float dx = __cosf(phi)*__sinf(theta);
    float dy = __sinf(phi)*__sinf(theta);
    float dz = __cosf(theta);

    // set particle stack
	d_g1.E[id] = 0.511f; // 511KeV
	d_g1.dx[id] = dx;
	d_g1.dy[id] = dy;
	d_g1.dz[id] = dz;
	d_g1.px[id] = x;
	d_g1.py[id] = y;
	d_g1.pz[id] = z;
	d_g1.tof[id] = 0.0f;
    d_g1.nCompton[id]=0;
    d_g1.nCompton_crystal[id]=0;
    d_g1.nPhotoelectric_crystal[id]=0;
	d_g1.endsimu[id] = 0;
   	d_g1.active[id] = 1;
    d_g1.crystalID[id]=-1;
    d_g1.Edeposit[id]=0;
   
	d_g2.E[id] = 0.511f;
	d_g2.dx[id] = -dx; // Back2back
	d_g2.dy[id] = -dy; //
	d_g2.dz[id] = -dz; //
	d_g2.px[id] = x;
	d_g2.py[id] = y;
	d_g2.pz[id] = z;
	d_g2.tof[id] = 0.0f;
    d_g2.nCompton[id]=0;
    d_g2.nCompton_crystal[id]=0;
    d_g2.nPhotoelectric_crystal[id]=0;
	d_g2.endsimu[id] = 0;
	d_g2.active[id] = 1;
    d_g2.crystalID[id]=-1;
    d_g2.Edeposit[id]=0;
    
}

///// Navigation ///////////////////////////////////////////////////////////////

// Photons - regular tracking
#define PHOTON_PHOTOELECTRIC 1
#define PHOTON_COMPTON 2
#define PHOTON_STEP_LIMITER 3
#define PHOTON_BOUNDARY_VOXEL 4
#define FLOAT_MAX 1e+37
#define EPS 1e-04
__global__ void kernel_navigation_regular(GPUParticleStack photons,
                                          GPUPhantom phantom,
                                          GPUPhantomMaterials materials,
										  float radius,
                                          int* d_ct_sim) {

    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    
    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;
   
    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = photons.px[id];
    position.y = photons.py[id];
    position.z = photons.pz[id];

    // Defined index phantom
    int4 index_phantom;
    float3 ivoxsize = vec3_inverse(phantom.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index

    // Read direction
    float3 direction;
    direction.x = photons.dx[id];
    direction.y = photons.dy[id];
    direction.z = photons.dz[id];

    // Get energy
    float energy = photons.E[id];
   
    // Get material
    unsigned short int mat = phantom.data[index_phantom.w];

    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLOAT_MAX;
    unsigned char next_discrete_process = 0; 
    //float interaction_distance;
    //float cross_section;

    // Photoelectric
    float cross_sectionpe = PhotoElec_CS(materials, mat, energy);
    float interaction_distancepe = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_sectionpe);
    if (interaction_distancepe < next_interaction_distance) {
       next_interaction_distance = interaction_distancepe;
       next_discrete_process = PHOTON_PHOTOELECTRIC;
    }

    // Compton
    float cross_sectioncpt = Compton_CS(materials, mat, energy);
    float interaction_distancecpt = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_sectioncpt);
    if (interaction_distancecpt < next_interaction_distance) {
       next_interaction_distance = interaction_distancecpt;
       next_discrete_process = PHOTON_COMPTON;
    }

    // Distance to the next voxel boundary (raycasting)
    float interaction_distancegeo = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    if (interaction_distancegeo < next_interaction_distance) {
      // overshoot the distance of 1 um to be inside the next voxel  
      next_interaction_distance = interaction_distancegeo;//+1.0e-06f;
      next_discrete_process = PHOTON_BOUNDARY_VOXEL;
    }
    
    //// Move particle //////////////////////////////////////////////////////
    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
    
    photons.px[id] = position.x;
    photons.py[id] = position.y;
    photons.pz[id] = position.z;
    photons.tof[id] += gpu_speed_of_light * next_interaction_distance;

    // Stop simulation if out of the phantom or out of scanner radius (-5 due to float issues)
    if ( position.x < EPS || position.x >= phantom.size_in_mm.x
         || position.y < EPS || position.y >= phantom.size_in_mm.y 
         || position.z < EPS || position.z >= phantom.size_in_mm.z
	 || sqrt((position.x-(phantom.size_in_mm.x/2))*(position.x-(phantom.size_in_mm.x/2))
	    + (position.y-(phantom.size_in_mm.y/2))*(position.y-(phantom.size_in_mm.y/2)))>radius - 5.f ) {
        photons.endsimu[id] = 1;                     // stop the simulation
        atomicAdd(d_ct_sim, 1);                      // count simulated particles
        return;
    }
    
    //// Resolve discrete processe //////////////////////////////////////////
    
    // Resolve discrete processes
    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
       PhotoElec_SampleSecondaries(photons, id, d_ct_sim);
       //printf("id %i PE\n", id);
    }
    
    if (next_discrete_process == PHOTON_COMPTON) {
       Compton_SampleSecondaries(photons, id, d_ct_sim);
       //printf("id %i Compton\n", id);
    }

}
#undef PHOTON_PHOTOELECTRIC
#undef PHOTON_COMPTON
#undef PHOTON_STEP_LIMITER
#undef PHOTON_BOUNDARY_VOXEL
#undef FLOAT_MAX
#undef EPS

#define INF 1.0e9f
__global__ void kernel_detection(GPUParticleStack photons,
                                          GPUScanner scanner, GPUPhantom phantom) {

    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    
    if (id >= photons.size) return;
    if (!photons.active[id]) return;
   
    float px=photons.px[id]-(phantom.size_in_mm.x/2);
	//------------------------------------------------------------------------------
	//CAUTION: Flipping in Y and Z to obtain BrainPET images in correct orientation
	//------------------------------------------------------------------------------
    float py=(phantom.size_in_mm.y/2) - photons.py[id];
    float pz=(phantom.size_in_mm.z/2) - photons.pz[id];
	photons.dy[id]=-photons.dy[id];
	photons.dz[id]=-photons.dz[id];

    float A = (scanner.cyl_radius)*(scanner.cyl_radius) - px*px - py*py; 
    float B = photons.dx[id]*photons.dx[id] + photons.dy[id]*photons.dy[id] ;
    float C = px*photons.dx[id] + py*photons.dy[id] ;
    
    float CdivB=C/B;
    float CdivB_square=CdivB*CdivB;
    float AdivB=A/B;
    float H=CdivB_square + AdivB;

    if(H<=0.0f) return;

    float p_intersec_x, p_intersec_y, p_intersec_z;
    float SQRT=sqrt(H);
    //use intersection in the moving direction of the photon
    float lambda=-CdivB+SQRT;
    if(lambda<0.0f)
        lambda=-CdivB-SQRT;
    if(lambda<0.0f) return;

    p_intersec_x= px+ photons.dx[id]*lambda;
    p_intersec_y= py+ photons.dy[id]*lambda;
    p_intersec_z= pz+ photons.dz[id]*lambda;

   // if gamma only hits the end planes, jump to next one
    if((p_intersec_z>scanner.cyl_halfheight)||(p_intersec_z<-scanner.cyl_halfheight)) return;
            
    const int blockid_cyl= (int) ((p_intersec_z/scanner.block_pitch)+(scanner.nblock)/2.0f);
    const int b_min=max(blockid_cyl-1,0);
    const int b_max=min(blockid_cyl+1,scanner.nblock-1);

    int cassid;
    float asin_arg=p_intersec_y/(scanner.cyl_radius);
    if(asin_arg>1.0f)
        asin_arg=1.0f;
    else if(asin_arg<-1.0f)
        asin_arg=-1.0f;

    if(p_intersec_x >=0.0f)
        cassid= (int) (0.5f + scanner.ncass*(gpu_halfpi-asinf(asin_arg))/gpu_twopi);               
    else
        cassid= (int) (0.5f + scanner.ncass*(gpu_pi+gpu_halfpi + asinf(asin_arg))/gpu_twopi);
    
    if(cassid==32)
        cassid=0;
    int c_min=max(cassid-1,0);
    int c_max=min(cassid+1,scanner.ncass-1);

    if(c_min==0 || c_max==scanner.ncass-1){
        c_min=0;
        c_max=scanner.ncass-1;
    }

    // find the detecting crystal
    float dmin=INF;
    int tempid=-1;
    for(int icass=c_min; icass<=c_max;icass++){
        for(int ib=b_min; ib<=b_max;ib++){
                  
            const int blockid=icass*scanner.nblock+ib;
            float distance=0.0f;
            distance= mc_hit_OBB(px, py, pz,               
                             photons.dx[id], photons.dy[id], photons.dz[id],
                             -scanner.halfsize.x,+scanner.halfsize.x,-scanner.halfsize.y,+scanner.halfsize.y,-scanner.halfsize.z,+scanner.halfsize.z,
                             scanner.pos[blockid].x, scanner.pos[blockid].y, scanner.pos[blockid].z,
                             scanner.v0[icass].x, scanner.v0[icass].y, scanner.v0[icass].z,
                             scanner.v1[icass].x, scanner.v1[icass].y, scanner.v1[icass].z,
                             scanner.v2[icass].x, scanner.v2[icass].y, scanner.v2[icass].z);
                
            if(distance>0 && distance<dmin){
                dmin=distance;
                tempid=blockid;
            }
        }
    }
          
    if(tempid!=-1){
        p_intersec_x=dmin*photons.dx[id] + px - scanner.pos[tempid].x;
        p_intersec_y=dmin*photons.dy[id] + py - scanner.pos[tempid].y; ;
        p_intersec_z=dmin*photons.dz[id] + pz - scanner.pos[tempid].z; ;
        int cassid=tempid/scanner.nblock; 
        float p_ta,p_ax;
        p_ta = p_intersec_x*scanner.v0[cassid].x + p_intersec_y*scanner.v0[cassid].y + p_intersec_z*scanner.v0[cassid].z;
        p_ax = p_intersec_x*scanner.v2[cassid].x + p_intersec_y*scanner.v2[cassid].y + p_intersec_z*scanner.v2[cassid].z;
        int ic_ax=min((int)(p_ax/scanner.cax_pitch+scanner.ncry_ax/2),scanner.ncry_ax-1);
        int ic_ta=min((int)(p_ta/scanner.cta_pitch+scanner.ncry_ta/2),scanner.ncry_ta-1);                
      
        photons.crystalID[id]=tempid*scanner.blocksize + ic_ax*scanner.ncry_ax+ic_ta;
        photons.Edeposit[id]=photons.E[id];        
        photons.tof[id] += gpu_speed_of_light * dmin; 
    }
}
#undef INF


#define INF 1.0e9f
#define PHOTON_PHOTOELECTRIC 1
#define PHOTON_COMPTON 2
#define PHOTON_BOUNDARY_VOXEL 4
#define FLOAT_MAX 1e+37
__global__ void kernel_detection_physics(GPUParticleStack photons,
                                          GPUScanner scanner, GPUPhantom phantom) {

   	//1:get the photon index
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    //2:check if the photon is still active(absorbed or below the energy window)
    if (id >= photons.size) return;
    if (!photons.active[id]) return;
    
    //3:Get the initial position of the photon when it leaves the phantom
    float3 position;
    position.x=photons.px[id]-(phantom.size_in_mm.x/2);

	//------------------------------------------------------------------------------
	//CAUTION: Flipping in Y and Z to obtain BrainPET images in correct orientation
	//------------------------------------------------------------------------------
    position.y=(phantom.size_in_mm.y/2) - photons.py[id];
    position.z=(phantom.size_in_mm.z/2) - photons.pz[id];
	photons.dy[id]=-photons.dy[id];
	photons.dz[id]=-photons.dz[id];

    //4: raytracing from the phantom to the detector
    float A = (scanner.cyl_radius)*(scanner.cyl_radius) - position.x*position.x - position.y*position.y; 
    float B = photons.dx[id]*photons.dx[id] + photons.dy[id]*photons.dy[id];
    float C = position.x*photons.dx[id] + position.y*photons.dy[id];
    
    float CdivB=C/B;
    float CdivB_square=CdivB*CdivB;
    float AdivB=A/B;
    float H=CdivB_square + AdivB;

    if(H<=0.0f) return;

    float p_intersec_x, p_intersec_y, p_intersec_z;
    float SQRT=sqrt(H);
    //use intersection in the moving direction of the photon
    float lambda=-CdivB+SQRT;
    if(lambda<0.0f)
        lambda=-CdivB-SQRT;
    if(lambda<0.0f) return;

    p_intersec_x= position.x + photons.dx[id]*lambda;
    p_intersec_y= position.y + photons.dy[id]*lambda;
    p_intersec_z= position.z + photons.dz[id]*lambda;

   //5: confirm the index of the cassette,block and the crystal according to the intersection position.
   // if gamma only hits the end planes, jump to next one
    if((p_intersec_z>scanner.cyl_halfheight)||(p_intersec_z<-scanner.cyl_halfheight)) return;
            
    const int blockid_cyl= (int) ((p_intersec_z/scanner.block_pitch)+(scanner.nblock)/2.0f);
    const int b_min=max(blockid_cyl-1,0);
    const int b_max=min(blockid_cyl+1,scanner.nblock-1);

    int cassid;
    float asin_arg=p_intersec_y/(scanner.cyl_radius);
    if(asin_arg>1.0f)
        asin_arg=1.0f;
    else if(asin_arg<-1.0f)
        asin_arg=-1.0f;

    if(p_intersec_x >=0.0f)
        cassid= (int) (0.5f + scanner.ncass*(gpu_halfpi-asinf(asin_arg))/gpu_twopi);               
    else
        cassid= (int) (0.5f + scanner.ncass*(gpu_pi+gpu_halfpi + asinf(asin_arg))/gpu_twopi);
    
    if(cassid==32)
        cassid=0;
    int c_min=max(cassid-1,0);
    int c_max=min(cassid+1,scanner.ncass-1);

    if(c_min==0 || c_max==scanner.ncass-1){
        c_min=0;
        c_max=scanner.ncass-1;
    }

    // find the detecting crystal
    float dmin=INF;
    int tempid=-1;
    for(int icass=c_min; icass<=c_max;icass++){
        for(int ib=b_min; ib<=b_max;ib++){
                  
            const int blockid=icass*scanner.nblock+ib;
            float distance=0.0f;
            distance= mc_hit_OBB(position.x, position.y, position.z,               
                             photons.dx[id], photons.dy[id], photons.dz[id],
                             -scanner.halfsize.x,+scanner.halfsize.x,-scanner.halfsize.y,+scanner.halfsize.y,-scanner.halfsize.z,+scanner.halfsize.z,
                             scanner.pos[blockid].x, scanner.pos[blockid].y, scanner.pos[blockid].z,
                             scanner.v0[icass].x, scanner.v0[icass].y, scanner.v0[icass].z,
                             scanner.v1[icass].x, scanner.v1[icass].y, scanner.v1[icass].z,
                             scanner.v2[icass].x, scanner.v2[icass].y, scanner.v2[icass].z);
                
            if(distance>0 && distance<dmin){
                dmin=distance;    
                tempid=blockid;
            }
        }
    }

    int icass=tempid/scanner.nblock; 
      
    if(tempid!=-1 ){
        p_intersec_x=dmin*photons.dx[id] + position.x - scanner.pos[tempid].x;
        p_intersec_y=dmin*photons.dy[id] + position.y - scanner.pos[tempid].y; 
        p_intersec_z=dmin*photons.dz[id] + position.z - scanner.pos[tempid].z; 
        photons.tof[id] += gpu_speed_of_light * dmin; 

        float3 position;
        float3 direction;

        position.x=  p_intersec_x*scanner.v0[icass].x + p_intersec_y*scanner.v0[icass].y + p_intersec_z*scanner.v0[icass].z;
        position.y=  p_intersec_x*scanner.v1[icass].x + p_intersec_y*scanner.v1[icass].y + p_intersec_z*scanner.v1[icass].z;
        position.z=  p_intersec_x*scanner.v2[icass].x + p_intersec_y*scanner.v2[icass].y + p_intersec_z*scanner.v2[icass].z;
        
        direction.x=  photons.dx[id] *scanner.v0[icass].x + photons.dy[id] *scanner.v0[icass].y + photons.dz[id] *scanner.v0[icass].z;
        direction.y=  photons.dx[id] *scanner.v1[icass].x + photons.dy[id] *scanner.v1[icass].y + photons.dz[id] *scanner.v1[icass].z;
        direction.z=  photons.dx[id] *scanner.v2[icass].x + photons.dy[id] *scanner.v2[icass].y + photons.dz[id] *scanner.v2[icass].z;

        //6:Get the energy
        // Get energy
        float energy = photons.E[id];

        //record the current block information
        float block_edeposit=0;//deposited energy for the current block
        float3 centroid;
        centroid.x=0;
        centroid.y=0;
        centroid.z=0;

        //record the previous block information
        float edeposit_firstblock = 0;//deposited energy in the previous block which the max deposited energy
        float3 centroid_firstblock;
        int id_firstblock;
        centroid_firstblock.x=0;
        centroid_firstblock.y=0;
        centroid_firstblock.z=0;
        bool active=true;
        //7: photon tracking in the detector until the photon is not active
        //loop; until the photon loses all its energy in PE or compton scattering;
        //if it goes to the next block, then the energy must greater than 400kev, otherwise even if it loses all
        //its enegy in the next block, it still less than the energy threshold.

        while(active){
            ////7.1 Find next discrete interaction ///////////////////////////////////////
            float energy_deposit=0;

            // Find next discrete interaction, total_dedx and next discrete intraction distance
            float next_interaction_distance =  FLOAT_MAX;
            unsigned char next_discrete_process = 0; 
            
            //7.1.1: calculate the interaction distance of photoelectric effect
            // Photoelectric
            float cross_sectionpe = PhotoElec_CS(scanner.mat, 0, energy);
            float interaction_distancepe = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                             cross_sectionpe);
            if (interaction_distancepe < next_interaction_distance) {
               next_interaction_distance = interaction_distancepe;
               next_discrete_process = PHOTON_PHOTOELECTRIC;
            }
            //7.1.2: calculate the interaction distance of compton effect
            // Compton
            float cross_sectioncpt = Compton_CS(scanner.mat, 0, energy);
            float interaction_distancecpt = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                             cross_sectioncpt);
            if (interaction_distancecpt < next_interaction_distance) {
               next_interaction_distance = interaction_distancecpt;
               next_discrete_process = PHOTON_COMPTON;
            }

            //7.1.3 calculate the interaction distance to the next block
            // Distance to the next voxel boundary (raycasting)
            float interaction_distancegeo = get_boundary_block_by_raycasting(position, 
                                                                direction, scanner.halfsize);
            if (interaction_distancegeo < next_interaction_distance) {
              // overshoot the distance of 1 um to be inside the next voxel  
              next_interaction_distance = interaction_distancegeo+1.0e-06f;
              next_discrete_process = PHOTON_BOUNDARY_VOXEL;
              position.x += direction.x * next_interaction_distance;
              position.y += direction.y * next_interaction_distance;
              position.z += direction.z * next_interaction_distance;
              photons.tof[id] += gpu_speed_of_light * next_interaction_distance;
              //with hit infor
//              if(energy<0.000001f)
              //without hit infor
            	if(energy<0.4f)
              {
            	  active=false;
//            	  printf("the inactive energy is: %f Mev\n",energy);

              }

              else{

                  id_firstblock=tempid;
                  float distance;             
                  distance=Move_To_Next_Block(position, direction, tempid, scanner, dmin, id);
                  if(tempid==-1)        
                    active=false;
                  //if the photon moves to the next block,all the informations of the current block information will be set zero 
                  else{
                      next_interaction_distance=distance+1.0e-06f;
                      if(block_edeposit>edeposit_firstblock){
                        edeposit_firstblock=block_edeposit;
                        centroid_firstblock.x=centroid.x;
                        centroid_firstblock.y=centroid.y;
                        centroid_firstblock.z=centroid.z;
                      }
                      centroid.x=0;
                      centroid.y=0;
                      centroid.z=0;
                      block_edeposit=0;
                  }
              }//end of else
            }
            //7.2 move the particle according to 7.1
            //// Move particle //////////////////////////////////////////////////////
            position.x += direction.x * next_interaction_distance;
            position.y += direction.y * next_interaction_distance;
            position.z += direction.z * next_interaction_distance;
            photons.tof[id] += gpu_speed_of_light * next_interaction_distance;
            

            //// Resolve discrete processe //////////////////////////////////////////
            
            // Resolve discrete processes
            if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
               energy_deposit=energy; 
               energy=0.0f;
               photons.nPhotoelectric_crystal[id]++;
               active=false;
            }
            
            if (next_discrete_process == PHOTON_COMPTON) {
                float energy_before=energy;            
                Compton_SampleSecondaries_Detector(photons, id, direction, energy, active);
                energy_deposit=energy_before-energy;
                photons.nCompton_crystal[id]++;
            }

            //7.3: update the position and energy information   
            block_edeposit+=energy_deposit;
            centroid.x+=energy_deposit*position.x;
            centroid.y+=energy_deposit*position.y;
            centroid.z+=energy_deposit*position.z;
        }//just out the tracking loop

        //8: compare which block get the max deposited energy and confirm the block and crystal and the energy 
        if(block_edeposit>edeposit_firstblock){                
            centroid.x/=block_edeposit;
            centroid.y/=block_edeposit;
            centroid.z/=block_edeposit;
            int ic_ax=min((int)(centroid.z/scanner.cax_pitch+scanner.ncry_ax/2),scanner.ncry_ax-1);
            int ic_ta=min((int)(centroid.x/scanner.cta_pitch+scanner.ncry_ta/2),scanner.ncry_ta-1);                
            photons.Edeposit[id]=block_edeposit;
            photons.crystalID[id]=tempid*scanner.blocksize + ic_ax*scanner.ncry_ax+ic_ta;
        } 
        else{
            centroid_firstblock.x/=edeposit_firstblock;
            centroid_firstblock.y/=edeposit_firstblock;
            centroid_firstblock.z/=edeposit_firstblock;
            int ic_ax=min((int)(centroid_firstblock.z/scanner.cax_pitch+scanner.ncry_ax/2),scanner.ncry_ax-1);
            int ic_ta=min((int)(centroid_firstblock.x/scanner.cta_pitch+scanner.ncry_ta/2),scanner.ncry_ta-1);                
            photons.Edeposit[id]=edeposit_firstblock;
            photons.crystalID[id]=id_firstblock*scanner.blocksize + ic_ax*scanner.ncry_ax+ic_ta;
        }
    }
}
#undef INF
#undef PHOTON_PHOTOELECTRIC
#undef PHOTON_COMPTON
#undef PHOTON_BOUNDARY_VOXEL
#undef FLOAT_MAX


