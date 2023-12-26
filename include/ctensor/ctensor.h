
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
#include <stdint.h>

typedef float ctensor_data_t;

typedef struct {
    size_t          size;
    ctensor_data_t  *data;
} CTensor_s;

typedef void (*CTensor_Layer_cb)(void *);

typedef struct _layer_s {
    // Models are really just linked lists
    // with callbacks, and hyperparameters.
    struct _layer_s     *next;
    struct _layer_s     *prev;
    // Forward-pass layer callback function.
    CTensor_Layer_cb    fwd;
    // Backprop-pass layer callback function.
    CTensor_Layer_cb    bckp;
    // Cleanup (dealloc internal(_grad)).
    CTensor_Layer_cb    del;
    // Pointer to the 'prev' layer's 'in'.
    CTensor_s           *in;
    // Pointer to allocated 'out' Tensor.
    CTensor_s           *out;
    /*  Internal variables and/or state.
     *
     *  It's the layer's responsability to allocate and
     *  deallocate this variable, along with its contents.
     *
     *  Implementations of 'internal' are not standarized
     *  as the Model Abstraction API will never interact with
     *  them; and just serve as a way to pass variables
     *  that the layer needs without forcing this to be a
     *  single Tensor/variable. */
    void                *internal;
    /*  Gradient of each 'in' element,
     *  with respect to the loss function.
     *  Gradient which will be backpropagated to
     *  the 'prev' layer. */
    CTensor_s           *loss_grad;
    /*  Gradient of each internal element, exported as a
     *  tensor so that the Model Abstraction API can do
     *  autograd.
     *
     *  It's the layer's responsability to allocate and
     *  deallocate this gradient Tensor.
     *
     *  'internal_grad' being NULL at training epochs
     *  will be interpreted as this layer not having
     *  any "trainable" variables. */
    CTensor_s           *internal_grad;
} CTensor_Layer_s;

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
CTensor_s *ctensor_relu_b(CTensor_s *in, CTensor_s *out, CTensor_s *loss_grad);

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
 *  @param seed - Seed for the PRNG.
 *
 *  @return - New allocated tensor
 *  containing the random numbers.
*/
CTensor_s *ctensor_randu(size_t size, uint64_t seed);

/*
 *  Generates n-size random numbers
 *  sampled from a normal distribution.
 *  
 *  Uses Blackman's and Vigna's xoshiro128+
 *  and the Marsaglia polar method.
 *
 *  @param size - numbers to be generated.
 *  @param seed - Seed for the PRNG.
 *
 *  @return - New allocated tensor
 *  containing the random numbers.
*/
CTensor_s *ctensor_randn(size_t size, uint64_t seed);

/*
 *  Initialize weights using the Xavier-He initialization
 *  method.
 *
 *  @param in_size - Number of nodes in previous layer.
 *  @param out_size - Number of nodes in the current
 *  layer.
 *
 *  @return - Returns a tensor with size out_size x in_size.
*/
CTensor_s *ctensor_xavier_he_init(size_t in_size, size_t out_size, uint64_t seed);

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
CTensor_s *ctensor_fcl_fwd(CTensor_s *in, CTensor_s *kernel, CTensor_s *bias, CTensor_s *out);

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
                            CTensor_s *kernel_grad, CTensor_s *bias_grad);