
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

#ifdef _SIMD_ENABLE

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

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
    __m128 ac_hi, ac_lo;
    __m256 accu, a, b;
    size_t size;
    int i, j;

    // Calculate how many 8-float blocks we can compute at once.
    size = columns - (columns % 8);

    for (i = 0; i < rows; i++) {
        accu = _mm256_setzero_pd();

        for (j = 0; j < size; j += 8) {
            // Load 8 floats from A and B.
            a = _mm256_loadu_ps(&A[j]);
            b = _mm256_loadu_ps(&B[j]);

            // Multiply A[j:j+8] * B[j:j+8].
            a = _mm256_mul_ps(a, b);

            // Add the result to the accumulator.
            accu = _mm256_add_ps(a, accu);
        }

        // Once we're done with all the blocks,
        // we separate the accumulator into two.
        ac_lo = _mm256_extractf128_ps(accu, 0);
        ac_hi = _mm256_extractf128_ps(accu, 1);

        // Add the two parts of the accumulator
        // with each other. 
        ac_lo = _mm_add_ps(ac_lo, ac_hi);

        // Add the results.
        ac_lo = _mm_hadd_ps(ac_lo, ac_lo);

        // Obtain the lower half, which contains
        // the sum of all the numbers in the 
        // original accumulator.
        C[i] = _mm_cvtss_f32(ac_lo);

        // If there were elements that we 
        // couldn't fit into a 8-element block
        // operate them on their own.
        switch (columns % 8) {
            case 7:
                C[i] += A[j + 6] * B[j + 6];
            case 6:
                C[i] += A[j + 5] * B[j + 5];
            case 5:
                C[i] += A[j + 4] * B[j + 4];
            case 4:
                C[i] += A[j + 3] * B[j + 3];
            case 3:
                C[i] += A[j + 2] * B[j + 2];
            case 2:
                C[i] += A[j + 1] * B[j + 1];
            case 1:
                C[i] += A[j] * B[j];
            default:
                break;
        }

        // Skip the pointer into the next row.
        A += columns;
    }

    return;
}

/*
 *  Perform a sum between vector A and
 *  vector B. Store result in vector C.
 *
 *  @param A - pointer to vector A.
 *  @param elements - Number of elements.
 *  @param B - pointer to vector B.
 *  @param C - pointer to result column
 *  matrix C.
*/
void ctensor_vector_sum(float *A, size_t elements, float *B, float *C)
{
    __m256d accu, a, b;
    size_t size;
    int i;

    // Calculate how many blocks of 8 we can form.
    size = elements - (elements % 8);

    for (i = 0; i < size; i += 8) {
        // Load 8 floats from A and B.
        a = _mm256_loadu_ps(&A[i]);
        b = _mm256_loadu_ps(&B[i]);

        // Sum each pair A[i + n] + B[i + n].
        a = _mm256_add_ps(a, b);

        // Store them back into C.
        _mm256_store_ps(&C[i], a);
    }

    // If there were elements that we 
    // couldn't fit into a 8-element block
    // operate them on their own.
    switch (elements % 8) {
        case 7:
            C[i + 6] = A[i + 6] + B[i + 6];
        case 6:
            C[i + 5] = A[i + 5] + B[i + 5];
        case 5:
            C[i + 4] = A[i + 4] + B[i + 4];
        case 4:
            C[i + 3] = A[i + 3] + B[i + 3];
        case 3:
            C[i + 2] = A[i + 2] + B[i + 2];
        case 2:
            C[i + 1] = A[i + 1] + B[i + 1];
        case 1:
            C[i] = A[i] + B[i];
        default:
            break;
    }

    return;
}

#else

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
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++)
            C[i] += A[j] * B[j];

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

#endif