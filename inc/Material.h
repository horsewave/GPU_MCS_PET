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

#ifndef MATERIAL_H
#define MATERIAL_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include "structure.h"


///// Material Data-Base ////////////////////////////////////////////////////////////

class Material {
    public:
        Material();
        std::vector<std::string> mixture_Z;//full name of elements like:Oxygen,Carbon....
        std::vector<float> mixture_f; //the mass fraction of each elements;
        //std::vector<unsigned short int> mixture_n; // TODO Add compound definition
        std::string name;//material name, like Air, Water....
        float density;//material density;
        unsigned short int nb_elements;//compositioning element numbers;
};



#endif
