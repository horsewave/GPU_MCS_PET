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

#ifndef MATERIALDATABASE_H
#define MATERIALDATABASE_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include "structure.h"
#include "Material.h"



class MaterialDataBase {
    public:
        MaterialDataBase();
        void load_materials(std::string);
        void load_elements(std::string);

        std::map<std::string, Material> materials_database;// map for all the materials 
        std::map<std::string, unsigned short int>  elements_Z;//map for atomic number Z
        std::map<std::string, float> elements_A;//map for mass number  

    private:
        void skip_comment(std::istream &);
        std::string remove_white_space(std::string txt);

        std::string read_element_name(std::string);
        int read_element_Z(std::string);
        float read_element_A(std::string);
        
        std::string read_material_name(std::string);
        float read_material_density(std::string);
        unsigned short int read_material_nb_elements(std::string);
        std::string read_material_element(std::string);
        float read_material_fraction(std::string);
};



#endif
