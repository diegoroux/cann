
/*
 *  Header file for CTensor.
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

typedef float ctensor_data_t;

typedef struct {
    size_t          size;
    ctensor_data_t  *data;
} CTensor_s;

/*
 *  ReLU (Rectified Linear Unit) function.
 *  Applies ReLU element-wise to the provided
 *  Tensor.
 *
 *  @params in - Tensor input.
 *  @param out - Tensor outupt or NULL.
 *  If NULL, a new tensor will be allocated.
 *
 *  @return - Tensor output pointer.
*/
CTensor_s *ctensor_relu(CTensor_s *in, CTensor_s *out);

/*
 *  First order partial derivative of the
 *  ReLU (Rectified Linear Unit) function.
 *  Calculates the local gradient of
 *  each input.
 *
 *  @params in - Tensor input.
 *  @param out - Tensor output or NULL.
 *  If NULL, a new tensor will be allocated.
 *
 *  @return - Tensor output pointer.
*/
CTensor_s *ctensor_relu_b(CTensor_s *in, CTensor_s *out);

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
void ctensor_mv_dot_product(float *A, size_t rows, size_t columns, float *B, float *C);

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
void ctensor_vector_sum(float *A, size_t elements, float *B, float *C);

/*
 *  Allocate a new tensor.
 *
 *  @param size - Size of the new tensor.
 *
 *  @return - New allocated tensor pointer.
*/
CTensor_s *ctensor_new_tensor(size_t size);

/*
 *  De-allocate tensor.
 *
 *  @param tensor - Pointer to the tensor struct
 *  to be de-allocated.
*/
void ctensor_destroy_tensor(CTensor_s *tensor);

/*
 *  Generates n-size random numbers
 *  sampled from a uniform distribution
 *  within the range [0, 1).
 *  
 *  Uses Blackman's and Vigna's xoshiro128+.
 *
 *  @param size - numbers to be generated
 *
 *  @return - New allocated tensor
 *  containing the random numbers.
*/
CTensor_s *ctensor_randu(size_t size);

/*
 *  Generates n-size random numbers
 *  sampled from a normal distribution.
 *  
 *  Uses Blackman's and Vigna's xoshiro128+.
 *
 *  @param size - numbers to be generated
 *
 *  @return - New allocated tensor
 *  containing the random numbers.
*/
CTensor_s *ctensor_randn(size_t size);