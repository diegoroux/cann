
/*
 *  Copyright (C) 2023 Diego Roux
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, version 3 of the License.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include "ann_internal.h"

#include <stdio.h>

void matrix_product(matrix m1, size_t rows, size_t columns, array m2, array m3)
{
    for (int i = 0; i < rows; i++) {
        m3[i] = 0.00;
        for (int j = 0; j < columns; j++) {
            m3[i] += m2[j] * m1[i][j];
        }
    }

    return;
}

void matrix_addition(array m1, array m2, array m3, size_t rows)
{
    for (int i = 0; i < rows; i++)
        m3[i] = m1[i] + m2[i];

    return;
}

double sigmoid(double x)
{
    return 1/(1 + exp(-x));   
}