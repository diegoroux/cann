
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

typedef float ctensor_data_t;

typedef struct {
    size_t          size;
    ctensor_data_t  data;
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