
/*
 *  Layer activation functions for CTensor.
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

#include <ctensor/ctensor.h>

#include <stddef.h>
#include <stdlib.h>

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
CTensor_s *ctensor_relu(CTensor_s *in, CTensor_s *out)
{
    ctensor_data_t *d_in, *d_out;
    int i;

    // Allocate a new tensor.
    if (out == NULL) {
        out = ctensor_new_tensor(in->size);

        if (out == NULL)
            return NULL;
    }

    d_in = in->data;
    d_out = out->data;

    out->size = in->size;

    // Apply ReLU, max(0, d_in[i]);
    for (i = 0; i < in->size; i++)
        d_out[i] = (d_in[i] < 0) ? 0 : d_in[i];
    
    return out;
}

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
CTensor_s *ctensor_relu_b(CTensor_s *in, CTensor_s *out)
{
    ctensor_data_t *d_in, *d_out;
    int i;

    // Allocate a new tensor.
    if (out == NULL) {
        out = ctensor_new_tensor(in->size);

        if (out == NULL)
            return NULL;
    }

    d_in = in->data;
    d_out = out->data;

    // As well return the partial derivative for each
    // element, both tensors have the same size.
    out->size = in->size;

    // The derivative at point d_in[i] for ReLU (max(0, x))
    // equals 1, for all d_in[i] > 0.
    // And 0 for everything else. 
    for (i = 0; i < in->size; i++)
        d_out[i] = (d_in[i] <= 0) ? 0 : 1;
    
    return out;
}