
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

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t  size;
    void    *mem;
    size_t  used;
} mempool_ctx;

typedef double **matrix;
typedef double *array;

typedef struct {
    size_t      no_input_nodes;
    size_t      no_hidden_layers;
    size_t      no_hidden_nodes;
    size_t      no_output_nodes;
    mempool_ctx mempool;
    matrix      *weight_matrix;
    array       *bias_array;
    array       *hidden_nodes;
    array       output_nodes;
} ann_ctx;

// math.c
void matrix_product(matrix m1, size_t rows, size_t columns, array m2, array m3);
void matrix_addition(array m1, array m2, array m3, size_t rows);
double sigmoid(double x);

// mem.c
void mempool_init(mempool_ctx *ctx, size_t size);
void *mempool_alloc(mempool_ctx *ctx, size_t size);
void mempool_free(mempool_ctx *ctx);

//utils.c
matrix alloc_matrix(mempool_ctx *ctx, size_t rows, size_t columns);