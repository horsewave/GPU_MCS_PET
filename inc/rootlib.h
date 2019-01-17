#ifndef ROOTLIB_H
#define ROOTLIB_H
#include <iostream>
#include <string>
// #include "TFile.h"
// #include "TTree.h"
using namespace std;

// class TMapFile;
// class TSplineFit;
class TFile;
class TH2D;
class TH3D;
class TTree;

class RootManager{
    
  
    public:
        RootManager(string,string);
        ~RootManager();

     static RootManager* Manager(string filename="rootfile.root",string option="RECREATE")
   {
     if (!Rootmanager)
       {
         cout<<"RootManager created"<<endl;  
             Rootmanager = new  RootManager(filename,option);
       }
     return Rootmanager;
   }

   

    void CreateHisto(string dir, string name, string title,int nbin, double min, double max,string option="");
    void CreateHisto(string dir, string name, string title,int nbinx, double xmin, double xmax, int nbiny, double ymin, double ymax,string option="");
    void CreateHisto(string dir, string name, string title,
                   int nbinx, double xmin, double xmax,
                   int nbiny, double ymin, double ymax,
                   int nbinz, double zmin, double zmax);

    void CreateHistoF(string dir, string name, string title,
                   int nbinx, double xmin, double xmax,
                   int nbiny, double ymin, double ymax,
                   int nbinz, double zmin, double zmax);


    void FillHisto(string dir, string name, double x, double weight);
    void FillHisto(string dir, string name, double x, double y, double weight);
    void FillHisto(string dir, string name, double x, double y, double z, double weight);
    
    void FillHistoTab(string dir, string name, int xdim, int ydim, int zdim, double *weight);
    void FillHistoTab(string dir, string name, int xdim, double *weight);
    
    
    void Cd(string dir);
    
    void WriteHisto(string dir, string name);
//         
private:
    static RootManager* Rootmanager;
    TFile *rootfile;
    TTree *tree;
};

#endif