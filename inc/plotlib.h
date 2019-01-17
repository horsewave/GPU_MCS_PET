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

#ifndef PLOTLIB_H
#define PLOTLIB_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string>

#define INF 1.0e30f

class GNUPlot {
    public:
        GNUPlot();
        void energy_hist(float*, int, int);
        void plot3(float*, float*, float*, int, std::string);
        void plot2(float*, float*, int, std::string);
        void plot(float *, int, std::string);

};







#endif
