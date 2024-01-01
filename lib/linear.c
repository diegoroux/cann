
/*
 *  Linear algebra functions.
 *  Copyright (C) 2023 Diego Roux
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation, version 3 of the License.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stddef.h>

/*
 *  Dot-product against a column matrix.
 *
 *  B shall always be of size columns x 1.
 *  
 *  @param A - Pointer to the matrix
 *  @param rows - Number of A's rows.
 *  @param columns - Number of A's columns.
 *  @param B - Pointer to the B column matrix.
 *  @param C - Pointer to where the result of
 *  the dot product will be stored.
*/
void ctensor_mv_dot_product(float *A, size_t rows, size_t columns, float *B, float *C)
{
    float c;
    int i, j;

    for (i = 0; i < rows; i++) {
        c = 0.00;

        for (j = 0; j < columns; j++)
            c += A[j] * B[j];

        C[i] = c;

        A += columns;
    }

    return;
}

/*
 *  Perform a sum between vector A and vector
 *  B. Store result in vector C.
 *
 *  @param A - pointer to vector A.
 *  @param elements - Number of elements.
 *  @param B - pointer to vector B.
 *  @param C - pointer to result column
 *  matrix C.
*/
void ctensor_vector_sum(float *A, size_t elements, float *B, float *C)
{
    int i;

    for (i = 0; i < elements; i++)
        C[i] = A[i] + B[i];

    return;
}

/*
 *  Perform a multiplication between vector A and
 *  scalar alpha. Store result in vector B.
 *
 *  @param A - pointer to vector A.
 *  @param elements - Number of elements.
 *  @param alpha - Scalar alpha.
 *  @param B - pointer to vector result
*/
void ctensor_sv_mult(float *A, size_t elements, float alpha, float *B)
{
    int i;

    for (i = 0; i < elements; i++)
        B[i] = alpha * A[i];

    return;
}
