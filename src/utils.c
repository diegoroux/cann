
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

#include "ann_internal.h"

matrix alloc_matrix(mempool_ctx *ctx, size_t rows, size_t columns)
{
    matrix matrix;

    matrix = mempool_alloc(ctx, rows * sizeof(array));
    matrix[0] = mempool_alloc(ctx, rows * columns * sizeof(double));

    for (int i = 1; i < rows; i++)
        matrix[i] = matrix[0] + (i * columns);

    return matrix;
}