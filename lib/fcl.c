
/*
 *  Fully-connected layer (FCL) implementation for CTensor.
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

#include <stdlib.h>

typedef struct {
    CTensor_s   *kernel;
    CTensor_s   *bias;
} _fcl_s;

void ctensor_fcl_fwd(CTensor_Layer_s *layer);
void ctensor_fcl_bckp(CTensor_Layer_s *layer);
void ctensor_fcl_update(CTensor_Layer_s *layer);
void ctensor_fcl_del(CTensor_Layer_s *layer);

/*
 *  FCL initial layer function.
 *  Fills all the layer information for the
 *  Model Abstraction API. As defined in 
 *  the documentation.
 *
 *  @param layer - Pointer of the current
 *  layer "object" to be filled.
*/
void ctensor_fcl_init(CTensor_Layer_s *layer)
{
    _fcl_s *data;

    // Set all callbacks.
    layer->fwd = (CTensor_Layer_cb)ctensor_fcl_fwd;
    layer->bckp = (CTensor_Layer_cb)ctensor_fcl_bckp;
    layer->update = (CTensor_Layer_cb)ctensor_fcl_update;
    layer->del = (CTensor_Layer_cb)ctensor_fcl_del;

    // Allocate internal state.
    data = (_fcl_s *)malloc(sizeof(_fcl_s));
    layer->internal = (void *)data;

    if (data == NULL)
        return;

    // Allocate the Tensor for the weights.
    data->kernel = ctensor_new_tensor(layer->out->size * layer->in->size);

    if (data->kernel == NULL)
        return;

    // Allocate the Tensor for the bias.
    data->bias = ctensor_new_tensor(layer->out->size);

    if (data->bias == NULL)
        return;

    // The internal gradient is conformed by both the weights and the bias.
    layer->internal_grad =
            ctensor_new_tensor(data->kernel->size + data->bias->size);

    if (layer->internal_grad == NULL)
        return;

    return;
}

/*
 *  Implements the forward pass of the FCL.
 *
 *  Defined as O = W • X + B, where W is the
 *  weight matrix (out_size x in_size), X
 *  is the input data as a vector/column matrix,
 *  B is the bias data as a vector/column matrix,
 *  and O is the output (out_size x 1).
 *
 *  @param layer - Pointer of the current
 *  layer "object".
*/
void ctensor_fcl_fwd(CTensor_Layer_s *layer)
{
    CTensor_s *kernel, *bias, *in, *out;
    _fcl_s *data;

    in = layer->in;
    out = layer->out;

    data = (_fcl_s *)layer->internal;

    kernel = data->kernel;
    bias = data->bias;

    ctensor_mv_dot_product(kernel->data, out->size, in->size, in->data, out->data);
    ctensor_vector_sum(out->data, out->size, bias->data, out->data);

    return;
}

/*
 *  Implements the backprop pass of the FCL.
 *
 *  @param layer - Pointer of the current
 *  layer "object".
*/
void ctensor_fcl_bckp(CTensor_Layer_s *layer)
{
    ctensor_data_t *kernel_grad, *bias_grad, *loss_grad, *in_grad;
    ctensor_data_t *in_data, *kernel_data;
    size_t in_size, out_size;
    _fcl_s *data;
    int i, j;

    in_size = layer->in->size;
    out_size = layer->out->size;

    in_data = layer->in->data;

    data = (_fcl_s *)layer->internal;
    kernel_data = data->kernel->data;

    in_grad = layer->in_grad->data;
    loss_grad = layer->loss_grad->data;

    kernel_grad = layer->internal_grad->data;
    bias_grad = &kernel_grad[out_size * in_size];

    for (i = 0; i < in_size; i++)
        in_grad[i] = 0.00f;

    for (i = 0; i < out_size; i++) {
        for (j = 0; j < in_size; j++) {
            kernel_grad[i * out_size + j] = in_data[j] * loss_grad[i];
            in_grad[j] += kernel_data[i * out_size + j] * loss_grad[i];
        }

        bias_grad[i] = loss_grad[i];
    }

    return;
}

/*
 *  Learning callback function for FCL.
 *
 *  This function shall only be called once the
 *  Model Abstraction API or the user, has applied
 *  an optimization algorithm for the gradient
 *  descent (for internal_grad) and has stored the 
 *  results back in layer->internal_grad.
 *  
 *  @param layer - Pointer of the current
 *  layer "object".
*/
void ctensor_fcl_update(CTensor_Layer_s *layer)
{
    ctensor_data_t *kernel, *bias, *internal_grad;
    _fcl_s *data;
    int offset;

    internal_grad = layer->internal_grad->data;
    data = layer->internal;

    kernel = data->kernel->data;
    bias = data->bias->data;

    offset = data->kernel->size;

    ctensor_vector_sum(kernel, data->kernel->size, internal_grad, kernel);
    ctensor_vector_sum(bias, data->bias->size, &internal_grad[offset], bias);

    return;
}

/*
 *  Dealloc FCL Layer.
 *
 *  @param layer - Pointer of the current
 *  layer "object".
*/
void ctensor_fcl_del(CTensor_Layer_s *layer)
{
    _fcl_s *data;

    ctensor_destroy_tensor(layer->internal_grad);
    layer->internal_grad = NULL;

    data = layer->internal;

    ctensor_destroy_tensor(data->kernel);
    ctensor_destroy_tensor(data->bias);

    data->kernel = NULL;
    data->bias = NULL;

    free(data);

    return;
}