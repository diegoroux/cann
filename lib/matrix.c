
/*
 *  Matrix dot-product functions.
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

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#ifdef _SMID_ENABLE
/*
 *  SIMD-optimized dot-product.
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
void matrix_mult(double *A, size_t rows, size_t columns, double *B, double *C)
{
    __m128d ac_hi, ac_lo;
    __m256d accu, a, b;
    int size, i, j;

    // Calculate how many 4 double blocks we can compute at once.
    size = columns - (columns % 4);

    for (i = 0; i < rows; i++) {
        accu = _mm256_setzero_pd();

        for (j = 0; j < size; j += 4) {
            // Load 4 doubles from A and B.
            a = _mm256_loadu_pd(&A[j]);
            b = _mm256_loadu_pd(&B[j]);

            // Multiply A[j:j+4] * B[j:j+4].
            a = _mm256_mul_pd(a, b);

            // Add the result to the accumulator.
            accu = _mm256_add_pd(a, accu);
        }

        // Once we're done with all the blocks,
        // we separate the accumulator into two.
        ac_lo = _mm256_extractf128_pd(accu, 0);
        ac_hi = _mm256_extractf128_pd(accu, 1);

        // Add the two parts of the accumulator
        // with each other. 
        ac_lo = _mm_add_pd(ac_lo, ac_hi);

        // Add the results.
        ac_lo = _mm_hadd_pd(ac_lo, ac_lo);

        // Obtain the lower half, which contains
        // the sum of all the numbers in the 
        // original accumulator.
        C[i] = _mm_cvtsd_f64(ac_lo);

        // If there were elements that we 
        // couldn't fit into a 4-element block
        // operate them on their own.
        switch (columns % 4) {
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
void matrix_mult(double *A, size_t rows, size_t columns, double *B, double *C)
{
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            C[i] += A[j] * B[j];
        }

        A += columns;
    }

    return;
}
#endif