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

#include "../inc/MaterialDataBase.h"



//// MaterialDataBase class //////////////////////////////////////////////

MaterialDataBase::MaterialDataBase() {}

// Read material name
std::string MaterialDataBase::read_material_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read material density
float MaterialDataBase::read_material_density(std::string txt) {
    float res;
    // density
    txt = txt.substr(txt.find("d=")+2);
    std::string txt1 = txt.substr(0, txt.find(" "));
    std::stringstream(txt1) >> res;
    // unit
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    if (txt=="g/cm3")  return res *gram/cm3;
    if (txt=="mg/cm3") return res *mgram/cm3;
    return res;
}

// Read material number of elements
unsigned short int MaterialDataBase::read_material_nb_elements(std::string txt) {
    unsigned short int res;
    txt = txt.substr(txt.find("n=")+2);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read material element name
std::string MaterialDataBase::read_material_element(std::string txt) {
    txt = txt.substr(txt.find("name=")+5);
    txt = txt.substr(0, txt.find(";"));
    txt = remove_white_space(txt);
    return txt;
}

// Read material element fraction TODO Add compound definition
float MaterialDataBase::read_material_fraction(std::string txt) {
    float res;
    txt = txt.substr(txt.find("f=")+2);
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Load materials from data base
void MaterialDataBase::load_materials(std::string filename) {
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    float mat_f;
    unsigned short int i;
    unsigned short int ind = 0;

    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            Material mat;
            mat.name = read_material_name(line);
            mat.density = read_material_density(line);
            mat.nb_elements = read_material_nb_elements(line);

            i=0; while (i<mat.nb_elements) {
                std::getline(file, line);
                elt_name = read_material_element(line);
                mat_f = read_material_fraction(line);

                mat.mixture_Z.push_back(elt_name);
                mat.mixture_f.push_back(mat_f);

                ++i;
            }

            materials_database[mat.name] = mat;
            ++ind;
        
        } // if
        
    } // while

}

// Read element name
std::string MaterialDataBase::read_element_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read element Z
int MaterialDataBase::read_element_Z(std::string txt) {
    int res;
    txt = txt.substr(txt.find("Z=")+2);
    txt = txt.substr(0, txt.find("."));
    std::stringstream(txt) >> res;
    return res;
}

// Read element A
float MaterialDataBase::read_element_A(std::string txt) {
    float res;
    txt = txt.substr(txt.find("A=")+2);
    txt = txt.substr(0, txt.find("g/mole"));
    std::stringstream(txt) >> res;
    return res *gram/mole;
}


// Load elements from data file
void MaterialDataBase::load_elements(std::string filename) {
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    int elt_Z;
    float elt_A;

    while (file) {
        skip_comment(file);
        std::getline(file, line);

        if (file) {
            elt_name = read_element_name(line);
            elt_Z = read_element_Z(line);
            elt_A = read_element_A(line);

            elements_Z[elt_name] = elt_Z;
            elements_A[elt_name] = elt_A;
        }
        
    }
}

// Skip comment starting with "#"
void MaterialDataBase::skip_comment(std::istream & is) {
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

// Remove all white space
std::string MaterialDataBase::remove_white_space(std::string txt) {
    txt.erase(remove_if(txt.begin(), txt.end(), isspace), txt.end());
    return txt;
}

#endif
