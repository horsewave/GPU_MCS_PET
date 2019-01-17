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

#ifndef SCANNERLIB_CPP
#define SCANNERLIB_CPP

#include "../inc/scannerlib.h"




bool compare_time(Single single1,Single single2){
    if(single2.time>single1.time)
        return true;
    else
        return false;
}


// Set scanner geometry and calculate block position and orientation
Scanner::Scanner( float cyl_radius, float cyl_halfheight, float b_pitch, float transaxial_pitch, float axial_pitch,
                              unsigned short int n_cass, unsigned short int n_block, unsigned short int n_cry_ta, unsigned short int n_cry_ax,
                              float sizex, float sizey, float sizez, std::string material, float window, float lower_threshold) {
    ncass=n_cass;
    nblock=n_block;
    ncry_ta=n_cry_ta;
    ncry_ax=n_cry_ax;
    blocksize=ncry_ta*ncry_ax;
    casssize=blocksize*nblock;
    cta_pitch=transaxial_pitch;
    cax_pitch=axial_pitch;
    block_pitch=b_pitch;
    radius=cyl_radius;
    halfheight=cyl_halfheight;
    halfsize.x=sizex/2.0;
    halfsize.y=sizey/2.0;
    halfsize.z=sizez/2.0;
    coinc_window=window*1.e-9;
    lld=lower_threshold;
     
    pos= (my_float3 *) malloc(ncass*nblock*sizeof(my_float3));
    v0= (my_float3 *) malloc(ncass*sizeof(my_float3));
    v1= (my_float3 *) malloc(ncass*sizeof(my_float3));
    v2= (my_float3 *) malloc(ncass*sizeof(my_float3));
    mat_name=material;

    for(int icass=0; icass<ncass;icass++){

        const float result_cos= cos(-icass*M_PI*2.0/ncass);
        const float result_sin= sin(-icass*M_PI*2.0/ncass);

        v0[icass].x=  result_cos;
        v0[icass].y=  result_sin;
        v0[icass].z= 0.0f;

        v1[icass].x= -result_sin;
        v1[icass].y= result_cos;
        v1[icass].z= 0.0f;
       
        v2[icass].x= 0.0f;
        v2[icass].y= 0.0f;
        v2[icass].z= 1.0f;

        for(int ib=0; ib<nblock;ib++){
           
            const int blockid=icass*nblock+ib;
            
            pos[blockid].x = -(radius + halfsize.y) * result_sin;    
            pos[blockid].y = (radius + halfsize.y ) * result_cos;            
            pos[blockid].z = (ib - (nblock-1)/2.0f)*block_pitch;
        }
    }
 
}
GPUPhantomMaterials Scanner:: defineMaterial(MaterialDataBase db){

    GPUPhantomMaterials gpumat;
    
    // nb of materials
    gpumat.nb_materials = 1;
    gpumat.nb_elements = (unsigned short int*)malloc(sizeof(unsigned short int) 
                                                        * gpumat.nb_materials);
    gpumat.index = (unsigned short int*)malloc(sizeof(unsigned short int) 
                                                        * gpumat.nb_materials);

    int j;
    unsigned int access_index = 0;
    unsigned int fill_index = 0;
    std::string elt_name;
    Material cur_mat;


    // read mat from databse
    cur_mat = db.materials_database[mat_name];
    if (cur_mat.name == "") {
        printf("[ERROR] Material %s is not on your database\n", mat_name.c_str());
        exit(EXIT_FAILURE);
    }

    // get nb of elements
    gpumat.nb_elements[0] = cur_mat.nb_elements;

    // compute index
    gpumat.index[0] = access_index;
    access_index += cur_mat.nb_elements;
        

    // nb of total elements
    gpumat.nb_elements_total = access_index;
    gpumat.mixture = (unsigned short int*)malloc(sizeof(unsigned short int)*access_index);
    gpumat.atom_num_dens = (float*)malloc(sizeof(float)*access_index);

    // store mixture element and compute atomic density

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

    return gpumat;

}


GPUScanner Scanner::get_scanner_for_GPU(MaterialDataBase db){
    GPUScanner scan;
    scan.cyl_radius=radius;
    scan.cyl_halfheight=halfheight;
    scan.block_pitch=block_pitch;
    scan.cta_pitch=cta_pitch;
    scan.cax_pitch=cax_pitch;
    scan.ncass=ncass;
    scan.nblock=nblock;
    scan.ncry_ta=ncry_ta;
    scan.ncry_ax=ncry_ax;
    scan.blocksize=blocksize;
    scan.halfsize=halfsize;
    scan.pos= (my_float3 *) malloc(ncass*nblock*sizeof(my_float3));
    scan.v0= (my_float3 *) malloc(ncass*sizeof(my_float3));
    scan.v1= (my_float3 *) malloc(ncass*sizeof(my_float3));
    scan.v2= (my_float3 *) malloc(ncass*sizeof(my_float3));

    scan.pos=pos;
    scan.v0=v0;
    scan.v1=v1;
    scan.v2=v2;

    GPUPhantomMaterials gpumat=defineMaterial(db);
    scan.mat =gpumat;
    //scan.mat = defineMaterial(db);

    return scan;

}

std::list<Single> Scanner::create_singles_list(GPUParticleStack g1, GPUParticleStack g2, float tot_activity){
    
    std::list<Single> SinglesList; 
    Single asingle;
    srand(100);
    double time=0.0;
    for (int i=0; i<(int)g1.size;i++){
       double ran=rand()/(double)(RAND_MAX);
       time += -log(ran)*(1./tot_activity);        
        if(g1.crystalID[i]!=-1 && g1.E[i]>lld){
            asingle.id=g1.crystalID[i];
            asingle.time=time+1.e-9*(double)g1.tof[i];
            asingle.eventID=i;
            asingle.nCompton=g1.nCompton[i];
            SinglesList.push_back(asingle);
           }
        if(g2.crystalID[i]!=-1 && g2.E[i]>lld){
            asingle.id=g2.crystalID[i];
            asingle.time=time+1.e-9*(double)g2.tof[i];
            asingle.eventID=i;
            asingle.nCompton=g2.nCompton[i];
            SinglesList.push_back(asingle);
        }
    }

    return SinglesList;
}

void Scanner::process_to_singles(GPUParticleStack g1, GPUParticleStack g2, std::list<Single>* SinglesList, float tot_activity){
    
    Single asingle;
    int absorbed=0;
    int compton=0;

    double time;
    //1:the list is empty,set the time 0
    if(SinglesList->empty())
        time=0.0;
    //there are already other singles,get the latest time as the initial time.Then all the singles are created chronologically.
    else{
        time=SinglesList->back().time;
        printf("Start Time: %f \n", time);
    }

    for (int i=0; i<(int)g1.size;i++){
       
	double ran=rand()/(double)(RAND_MAX);
	if(ran<=0 || ran>1.0){
	  printf("%f  -  ",ran);  
	  ran=rand()/(double)(RAND_MAX);
	}
       	time += -log(ran)*(1./tot_activity); 
        if(g1.crystalID[i]!=-1 && g1.Edeposit[i]>lld){
            asingle.id=g1.crystalID[i];
            asingle.time=time+1.e-9*(double)g1.tof[i];
            asingle.eventID=i;
            asingle.nCompton=g1.nCompton[i];
            SinglesList->push_back(asingle);
            if(g1.nCompton[i]>0)
                compton++;;
           }
        if(g1.active[i]==0)
            absorbed++;
        if(g2.crystalID[i]!=-1 && g2.Edeposit[i]>lld){
            asingle.id=g2.crystalID[i];
            asingle.time=time+1.e-9*(double)g2.tof[i];
            asingle.eventID=i;
            asingle.nCompton=g2.nCompton[i];
            SinglesList->push_back(asingle);
            if(g2.nCompton[i]>0)
                compton++;
        }
        if(g2.active[i]==0)
            absorbed++;
    }
    printf("Number of Singles: %i, Number of scattered: %i, Number of absorbed: %i Rest: %i \n", 
            (int) SinglesList->size(), compton, absorbed, (int)g1.size*2-(int)SinglesList->size()-absorbed);
}


void Scanner::Get_hits_info(GPUParticleStack g1, GPUParticleStack g2)
{
	int sum_photoelectric=0;
	int sum_compton_crystal=0;
	int sum_compton_phantom=0;


    int count_j=0;
	for (int i=0; i<(int)g1.size;i++)
		{


			if(g1.crystalID[i]!=-1)
			{
				if(g1.nCompton_crystal[i]>0)
				{
//					printf("g1-eventID: %i, compton in Crystal: %i, \n",i,g1.nCompton_crystal[i]);
					count_j++;


				}
				if(g1.nPhotoelectric_crystal[i]>0)
				{
//					printf("g1-eventID: %i, PE in Crystal: %i, \n",i,g1.nPhotoelectric_crystal[i]);
				}



			}

			if(g2.crystalID[i]!=-1)
			{
				if(g2.nCompton_crystal[i]>0)
				{
//					printf("g2-eventID: %i, compton in Crystal: %i, \n",i,g2.nCompton_crystal[i]);

				}
				if(g2.nPhotoelectric_crystal[i]>0)
				{
//					printf("g2-eventID: %i, PE in Crystal: %i, \n",i,g2.nPhotoelectric_crystal[i]);
	//
				}



			}

			if(count_j>1000)
			{
				break;
			}

		}










	for (int i=0; i<(int)g1.size;i++)
	{


		if(g1.crystalID[i]!=-1)
		{
			if(g1.nCompton_crystal[i]>0)
			{
				sum_compton_crystal+=g1.nCompton_crystal[i];
				//sum_compton++;
			}
			if(g1.nPhotoelectric_crystal[i]>0)
			{
				//sum_photoelectric+=g1.nPhotoelectric_crystal[i];
				sum_photoelectric++;
			}

			if(g1.nPhotoelectric_crystal[i]>0||g1.nCompton_crystal[i]>0)
			{
				sum_compton_phantom+=g1.nCompton[i];
			}

		}

		if(g2.crystalID[i]!=-1)
		{
			if(g2.nCompton_crystal[i]>0)
			{
				sum_compton_crystal+=g2.nCompton_crystal[i];
//				sum_compton++;
			}
			if(g2.nPhotoelectric_crystal[i]>0)
			{
//				sum_photoelectric+=g2.nPhotoelectric_crystal[i];
				sum_photoelectric++;
			}

			if(g2.nPhotoelectric_crystal[i]>0||g2.nCompton_crystal[i]>0)
			{
				sum_compton_phantom+=g2.nCompton[i];
			}

		}

	}

	printf("Number of compton in Phantom: %i,Number of compton in Crystal: %i, Number of photoelectric in crystal: %i \n",sum_compton_phantom,sum_compton_crystal,sum_photoelectric);
}




void Scanner::save_coincidences(std::list<Single>* SinglesList, std::string fname, bool start, bool end, float* lorvec_scat, float* lorvec_true){


    std::list<Coincidence> CoincidenceList; 
    Coincidence acoinc;
    //sort Singles in time to simplify coincidence sorting
    SinglesList->sort(compare_time);
    std::list<Single>::iterator it_temp_end=SinglesList->end();
    if(!end)
        advance(it_temp_end,-10);
    //sort Singles into Coincidences
    for (std::list<Single>::iterator it=SinglesList->begin(); it != it_temp_end; it++){
        std::list<Single>::iterator it2=it;
        it2++;
        int nr_single=0;
        while(it2->time<it->time+coinc_window && it2!=it_temp_end){
            it2++;
            nr_single++;
        }
        //only one single in the opened window
        if(nr_single==1)  {    
            it2--;
            std::list<Single>::iterator it3=it2;
            it3++;
            // make sure it is not also in coincidence with a later single
            if(it3->time>it2->time+coinc_window && it3!= it_temp_end){    
                //write to Coincidence List
                    acoinc.one=*it;
                    acoinc.two=*it2;
                if(it->eventID == it2->eventID){
                    if(acoinc.one.nCompton==0 && acoinc.two.nCompton==0)
                        acoinc.type=1;
                    else
                        acoinc.type=2;
                }
                else
                    acoinc.type=3;
                CoincidenceList.push_back(acoinc);
                it=it2;
            }
            // there is a multiple coincidence, jump to next event that is not involved
            else{
                while(it3->time<it2->time+coinc_window && it3!=it_temp_end)
                    it3++;
                it=--it3;
            }
        } 
       // there is a multiple coincidence, jump to next event
       else if(nr_single>1)
           it=--it2;
    }

    SinglesList->erase(SinglesList->begin(),it_temp_end);

    int  blocksize=144;
    int casssize=blocksize*6;
    int ncrystals=casssize*32;
    int nrandoms=0;
    int nscatter=0;
    int rsector1,rsector2,i1,submodule1,submodule2,crystal1,crystal2;

    std::ofstream fout;
    std::string file_lm=fname + ".bin";
    if(start)
        fout.open(file_lm.c_str() );
    else
        fout.open(file_lm.c_str(), std::ios_base::app);

    for(std::list<Coincidence>::iterator it=CoincidenceList.begin(); it != CoincidenceList.end(); it++){
        
        int icry1=it->one.id;
        char flag=it->type;
        int icry2=it->two.id;

        if(it->type==3)
            nrandoms++;
        else if(it->type==2){
            nscatter++;
            if(lorvec_scat!=NULL){
                int rsector1 = icry1/casssize;
                int i1  = icry1 % casssize;
                int submodule1 = i1/ blocksize;
                int cry1 = i1 % blocksize;
                cry1=cry1+rsector1*blocksize+submodule1*blocksize*32;
                int rsector2 = icry2/casssize;
                i1  = icry2 % casssize;
                int submodule2 = i1/ blocksize;
                int cry2 = i1 % blocksize;
                cry2=cry2+rsector2*blocksize+submodule2*blocksize*32;
                int lorid;
                if(cry1>cry2)
                    lorid=(cry1-cry2-1)+cry2*ncrystals-(cry2*cry2+cry2)/2;
                else
                    lorid=(cry2-cry1-1)+cry1*ncrystals-(cry1*cry1+cry1)/2;   

                lorvec_scat[lorid]++;
            }
        }
        else{
            if(lorvec_true!=NULL){
                int rsector1 = icry1/casssize;
                int i1  = icry1 % casssize;
                int submodule1 = i1/ blocksize;
                int cry1 = i1 % blocksize;
                cry1=cry1+rsector1*blocksize+submodule1*blocksize*32;
                int rsector2 = icry2/casssize;
                i1  = icry2 % casssize;
                int submodule2 = i1/ blocksize;
                int cry2 = i1 % blocksize;
                cry2=cry2+rsector2*blocksize+submodule2*blocksize*32;
                int lorid;
                if(cry1>cry2)
                    lorid=(cry1-cry2-1)+cry2*ncrystals-(cry2*cry2+cry2)/2;
                else
                    lorid=(cry2-cry1-1)+cry1*ncrystals-(cry1*cry1+cry1)/2;  
 
                lorvec_true[lorid]++;
            }
        }

		fout.write((char *) &icry1, sizeof(int));
        fout.write((char *) &icry2, sizeof(int));
        fout.write((char *) &flag, sizeof(char));
    }
    fout.close();

    unsigned int nentries=CoincidenceList.size();   
    printf("Trues: %i, therein scattered: %i, Fraction: %.3f, Randoms: %i, Total %i, Fraction: %.3f \n",
            nentries-nrandoms, nscatter, nscatter*100./(float)(nentries-nrandoms),nrandoms,(int) nentries, nrandoms*100./(float)nentries);
}

void Scanner::save_coincidences_lor(std::list<Single>* SinglesList, std::string fname, bool start, bool end, float* lortrue, float* lorscat){


    std::list<Coincidence> CoincidenceList; 
    Coincidence acoinc;
    //sort Singles in time to simplify coincidence sorting
    SinglesList->sort(compare_time);
    std::list<Single>::iterator it_temp_end=SinglesList->end();
    if(!end)
        advance(it_temp_end,-10);
    //sort Singles into Coincidences
    for (std::list<Single>::iterator it=SinglesList->begin(); it != it_temp_end; it++){
        std::list<Single>::iterator it2=it;
        it2++;
        int nr_single=0;
        while(it2->time<it->time+coinc_window && it2!=it_temp_end){
            it2++;
            nr_single++;
        }
        //only one single in the opened window
        if(nr_single==1)  {    
            it2--;
            std::list<Single>::iterator it3=it2;
            it3++;
            // make sure it is not also in coincidence with a later single
            if(it3->time>it2->time+coinc_window && it3!= it_temp_end){    
                //write to Coincidence List
                    acoinc.one=*it;
                    acoinc.two=*it2;
                if(it->eventID == it2->eventID){
                    if(acoinc.one.nCompton==0 && acoinc.two.nCompton==0)
                        acoinc.type=1;
                    else
                        acoinc.type=2;
                }
                else
                    acoinc.type=3;
                CoincidenceList.push_back(acoinc);
                it=it2;
            }
            // there is a multiple coincidence, jump to next event that is not involved
            else{
                while(it3->time<it2->time+coinc_window && it3!=it_temp_end)
                    it3++;
                it=--it3;
            }
        } 
       // there is a multiple coincidence, jump to next event
       else if(nr_single>1)
           it=--it2;
    }

    SinglesList->erase(SinglesList->begin(),it_temp_end);

	// information for processing to LOR file
    int  blocksize=144;
    int casssize=blocksize*6;
    int ncrystals=casssize*32;
	int Nhead_EPM=32;
 	int addr;
  ////memory not free-Bo
 	short int *hplookup=(short int*) malloc(sizeof(short int)*Nhead_EPM*Nhead_EPM);
 	create_hplookup(hplookup);      

    int nrandoms=0;
    int nscatter=0;
  

    std::ofstream fout;
    std::string file_lm=fname + ".bin";
    if(start)
        fout.open(file_lm.c_str() );
    else
        fout.open(file_lm.c_str(), std::ios_base::app);

    for(std::list<Coincidence>::iterator it=CoincidenceList.begin(); it != CoincidenceList.end(); it++){
        
        int icry1=it->one.id;
        char flag=it->type;
        int icry2=it->two.id;

        if(it->type==3)
            nrandoms++;
        else if(it->type==2){
            nscatter++;
			// lorfile processing
            if(lorscat!=NULL){
        		int icass1 = icry1/casssize;
        		int i1  = icry1 % casssize;
        		int iblock1 = i1/ blocksize;
        		int i2 = i1 % blocksize;
				int ic_ax = i2 / 12; 
        		int ic_ta = i2 % 12;
				int ic1=143-(ic_ta*12+ic_ax);    
				int icass2 = icry2/casssize;
        		i1  = icry2 % casssize;
        		int iblock2 = i1/ blocksize;
        		i2 = i1 % blocksize;
				ic_ax = i2 / 12; 
        		ic_ta = i2 % 12;
				int ic2=143-(ic_ta*12+ic_ax);    

	 			addr=get_LOR_flat_address(hplookup,icass1,iblock1,ic1,icass2,iblock2,ic2);
				if(addr >0 && addr<304*864*864)
                	lorscat[addr]++;
            }
        }
        else{
            if(lortrue!=NULL){
        		int icass1 = icry1/casssize;
        		int i1  = icry1 % casssize;
        		int iblock1 = i1/ blocksize;
        		int i2 = i1 % blocksize;
				int ic_ax = i2 / 12; 
        		int ic_ta = i2 % 12;
				int ic1=143-(ic_ta*12+ic_ax);    
				int icass2 = icry2/casssize;
        		i1  = icry2 % casssize;
        		int iblock2 = i1/ blocksize;
        		i2 = i1 % blocksize;
				ic_ax = i2 / 12; 
        		ic_ta = i2 % 12;
				int ic2=143-(ic_ta*12+ic_ax);    

	 			addr=get_LOR_flat_address(hplookup,icass1,iblock1,ic1,icass2,iblock2,ic2);
				if(addr >0 && 304*864*864)
                	lortrue[addr]++;				
            }
        }

		fout.write((char *) &icry1, sizeof(int));
        fout.write((char *) &icry2, sizeof(int));
        fout.write((char *) &flag, sizeof(char));
    }
    fout.close();

    unsigned int nentries=CoincidenceList.size();   
    printf("Trues: %i, therein scattered: %i, Fraction: %.3f, Randoms: %i, Total %i, Fraction: %.3f \n",
            nentries-nrandoms, nscatter, nscatter*100./(float)(nentries-nrandoms),nrandoms,(int) nentries, nrandoms*100./(float)nentries);
    //free(hplookup);
}

void Scanner::create_hplookup(short int *hplookup){

    int hoff, a,b;
    int i, o, hb ,cntr;
    const int BrainPET_n_FOV = 19;     
    const int Nhead_EPM=32;    
    short int *hpair;
    int n_mpairs;

    //memory not free-Bo
    hpair=(short int*) malloc(sizeof(short int)*Nhead_EPM*(Nhead_EPM-1));
   // hplookup=(short int*) malloc(sizeof(short int)*Nhead_EPM*Nhead_EPM);

    hoff=(Nhead_EPM-BrainPET_n_FOV+1)/2;
    cntr=0;
    n_mpairs=Nhead_EPM*BrainPET_n_FOV/2;
    // check to be sure we don't exceed the maximum allowable (496=32*31/2);
    if (n_mpairs > 496){
        printf("Invalid head_fov..n_mpairs\n");
        exit(0);
    }
    if (BrainPET_n_FOV%2 == 0){
        printf("FOV should be odd number\n");
        exit(0);
    }
    for (i=0; i<Nhead_EPM; i++) {
        for (o=0; o<BrainPET_n_FOV; o++) {
              hb=i+o+hoff;
              if (hb < Nhead_EPM) {
                hpair[2*cntr]=i;
                hpair[2*cntr+1]=hb;
                cntr++;
              }
        }
    }
    if (cntr != n_mpairs){
        printf("init_lor_info(): You must have goofed up somewhere n_mpairs)\n");
        exit(0);
    }
    for (i=0; i<Nhead_EPM*Nhead_EPM; i++) 
        hplookup[i]=(short int)0;
    for (i=0; i<n_mpairs; i++) {
        a=hpair[2*i];
        b=hpair[2*i+1];
        hplookup[32*b+a]=(short int)(i+1);
        hplookup[32*a+b]=(short int)(-i-1);
    }
    //free(hpair);
}

int Scanner::get_LOR_flat_address(short int *hplookup, int h1, int b1, int c1,int h2, int b2,int c2){

    int a1; //adress of crystal_1
    int a2; //adress of crystal_2
    int ax; //tmp for switch
    int addr; //total adress of lor in lor-array

    int hp=hplookup[h1*32+h2];

    if (hp == 0) {
        return -1;
    }

    a1=144*b1+c1;
    a2=144*b2+c2;
    if (hp < 0) {
        hp=-hp;
        ax=a1;
        a1=a2;
        a2=ax;
    }

    //--- calculation of adress:
    addr=(144*6*144*6)*(hp-1)+a2*144*6+a1;
    return addr;

}

Scanner::~Scanner(){
    free(pos);
    free(v0);
    free(v1);
    free(v2);
}


#endif

