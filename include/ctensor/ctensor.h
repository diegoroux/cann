
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
    // Autograd update callback function.
    CTensor_Layer_cb    update;
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
    CTensor_s           *in_grad;
    /*  Gradient of each 'out' element,
     *  with respect to the loss function.
     *  Pointer to the 'next' in_grad. */
    CTensor_s           *loss_grad;
    /*  Gradient of each internal element, exported as a
     *  tensor so that the Model Abstraction API can do
     *  optimization.
     *
     *  It's the layer's responsability to allocate and
     *  deallocate this gradient Tensor.
     *
     *  'internal_grad' being NULL at training epochs
     *  will be interpreted as this layer not having
     *  any "trainable" variables. */
    CTensor_s           *internal_grad;
} CTensor_Layer_s;

typedef ctensor_data_t (*CTensor_Loss_cb)(void *, CTensor_s *);

typedef struct _loss_ls {
    // Models are really just linked lists
    // with callbacks, and hyperparameters.
    CTensor_Layer_s     *prev;
    // Forward-pass layer callback function.
    CTensor_Loss_cb     fwd;
    // Backprop-pass layer callback function.
    CTensor_Loss_cb     bckp;
    /*  Gradient of each 'in' element,
     *  with respect to the loss function.
     *  Gradient which will be backpropagated to
     *  the 'prev' layer. */
    CTensor_s           *in_grad;
} CTensor_Loss_s;

/*
 *  ReLU initial layer function.
 *  Fills all the layer information for the
 *  Model Abstraction API. As defined in 
 *  the documentation.
 *
 *  @param layer - Pointer of the current
 *  ReLU layer "object" to be filled.
*/
void ctensor_relu_init(CTensor_Layer_s *layer);

/*
 *  ReLU (Rectified Linear Unit) forward pass
 *  function, applies ReLU element-wise to
 *  the provided Tensor.
 *
 *  @params layer - Pointer to the current
 *  ReLU layer "object".
*/
void ctensor_relu_fwd(CTensor_Layer_s *layer);

/*
 *  First order partial derivative of the
 *  ReLU (Rectified Linear Unit) function.
 *  Calculates the local gradient of
 *  each input, and by chain rule, multiplies
 *  it by the loss gradient it receives.
 *
 *  @params layer - Pointer to the current
 *  ReLU layer "object".
*/
void ctensor_relu_bckp(CTensor_Layer_s *layer);

/*
 *  FCL initial layer function.
 *  Fills all the layer information for the
 *  Model Abstraction API. As defined in 
 *  the documentation.
 *
 *  @param layer - Pointer of the current
 *  layer "object" to be filled.
*/
void ctensor_fcl_init(CTensor_Layer_s *layer);

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
 *  ReLU layer "object" to be filled.
*/
void ctensor_fcl_fwd(CTensor_Layer_s *layer);

/*
 *  Implements the backprop pass of the FCL.
 *
 *  @param layer - Pointer of the current
 *  ReLU layer "object" to be filled.
*/
void ctensor_fcl_bckp(CTensor_Layer_s *layer);

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
void ctensor_fcl_update(CTensor_Layer_s *layer);

/*
 *  Dealloc FCL Layer.
 *
 *  @param layer - Pointer of the current
 *  layer "object".
*/
void ctensor_fcl_del(CTensor_Layer_s *layer);

/*
 *  Initializes the Loss Layer with fwd and bck
 *  callbacks.
 *
 *  @param layer - Current layer "object".
*/
void ctensor_mse_init(CTensor_Loss_s *layer);

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
 *  @param tensor - Tensor to be filled.
 *  @param seed - Seed for the PRNG.
*/
void ctensor_randu(CTensor_s *tensor, uint64_t seed);

/*
 *  Generates n-size random numbers
 *  sampled from a normal distribution.
 *  
 *  Uses Blackman's and Vigna's xoshiro128+
 *  and the Marsaglia polar method.
 *
 *  @param tensor - Tensor to be filled.
 *  @param seed - Seed for the PRNG.
*/
void ctensor_randn(CTensor_s *tensor, uint64_t seed);

/*
 *  Initialize weights using the Xavier-He initialization
 *  method.
 *
 *  @param tensor - Pointer to the Tensor to be initialization.
 *  @param in_size - Number of nodes in the previous layer.
 *  @param seed - Seed for the PRNG.
*/
void ctensor_xavier_he_init(CTensor_s *tensor, size_t in_size, uint64_t seed);