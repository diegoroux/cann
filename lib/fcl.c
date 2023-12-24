
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

/*
 *  Implements the forward pass of the FCL.
 *
 *  Defined as O = W • X + B, where W is the
 *  weight matrix (out_size x in_size), X
 *  is the input data as a vector/column matrix,
 *  B is the bias data as a vector/column matrix,
 *  and O is the output (out_size x 1).
 *
 *  @param in - Tensor coming in.
 *  @param kernel - Weight tensor.
 *  @param bias - Bias tensor.
 *  @param out - Out tensor.
 *
 *  @return - Pointer to the out tensor.
*/
CTensor_s *ctensor_fcl_fwd(CTensor_s *in, CTensor_s *kernel, CTensor_s *bias, CTensor_s *out)
{
    ctensor_mv_dot_product(kernel->data, out->size, in->size, in->data, out->data);
    ctensor_vector_sum(out->data, out->size, bias->data, out->data);

    return out;
}

/*
 *  Implements the backprop pass of the FCL.
 *
 *  @param in - Tensor coming in.
 *  @param kernel - Weight tensor.
 *  @param loss_grad - Gradient backpropagated.
 *  @param kernel_grad - Weight gradient with respect to
 *  the loss function.
 *  @param bias_grad - Bias gradient with respect to the
 *  loss function.
 *
 *  @return - Pointer to the tensor, contaning the gradient
 *  of the input with respect to the loss function.
*/
CTensor_s *ctensor_fcl_bckp(CTensor_s *in, CTensor_s *kernel, CTensor_s *loss_grad,
                            CTensor_s *kernel_grad, CTensor_s *bias_grad)
{
    CTensor_s *in_grad;
    int i, j;

    in_grad = ctensor_new_tensor(in->size);
    
    if (in_grad == NULL)
        return NULL;

    for (i = 0; i < in->size; i++)
        in_grad->data[i] = 0.00f;

    // TODO: Optimize this.
    for (i = 0; i < loss_grad->size; i++) {
        for (j = 0; j < in->size; j++) {
            kernel_grad->data[i * loss_grad->size + j] = in->data[j] * loss_grad->data[i];
            in_grad->data[j] += kernel->data[i * loss_grad->size + j] * loss_grad->data[i];
        }

        bias_grad->data[i] = loss_grad->data[i];   
    }

    return in_grad;
}