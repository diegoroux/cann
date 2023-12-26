
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

/*
 *  ReLU initial layer function.
 *  Fills all the layer information for the
 *  Model Abstraction API. As defined in 
 *  the documentation.
 *
 *  @param layer - Pointer of the current
 *  ReLU layer "object" to be filled.
*/
void ctensor_relu_init(CTensor_Layer_s *layer)
{
    // This layer doesn't have any trainable
    // variables, as such we do not implement
    // any 'update' nor 'del' methods.
    layer->fwd = ctensor_relu_fwd;
    layer->bckp = ctensor_relu_bckp;
    layer->update = NULL;
    layer->del = NULL;

    // This layer is not trainable.
    layer->internal = NULL;
    layer->internal_grad = NULL;

    return;
}

/*
 *  ReLU (Rectified Linear Unit) forward pass
 *  function, applies ReLU element-wise to
 *  the provided Tensor.
 *
 *  @params layer - Pointer to the current
 *  ReLU layer "object".
*/
void ctensor_relu_fwd(CTensor_Layer_s *layer)
{
    ctensor_data_t *in, *out;
    int i;

    in = layer->in->data;
    out = layer->out->data;

    // Apply ReLU, max(0, d_in[i]);
    for (i = 0; i < in->size; i++)
        out[i] = (in[i] < 0) ? 0 : in[i];

    return;
}

/*
 *  First order partial derivative of the
 *  ReLU (Rectified Linear Unit) function.
 *  Calculates the local gradient of
 *  each input, and by chain rule, multiplies
 *  it by the loss gradient it receives.
 *
 *  @params in - Tensor input.
 *  @param out - Tensor output or NULL.
 *  If NULL, a new tensor will be allocated.
 *  @param loss_grad - Tensor holding the gradient
 *  that's being backpropagated.
 *
 *  @return - Tensor output pointer.
*/
void ctensor_relu_bckp(CTensor_Layer_s *layer)
{
    ctensor_data_t *in, *out, *loss;
    int i;

    in = layer->in->data;
    out = layer->in_grad->data;
    loss = layer->loss_grad->data;

    // The derivative at point d_in[i] for ReLU (max(0, x))
    // equals 1, for all d_in[i] > 0.
    // And 0 for everything else. 
    for (i = 0; i < in->size; i++)
        out[i] = (in[i] <= 0) ? 0 : loss[i];

    return;
}