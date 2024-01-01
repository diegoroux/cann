
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

struct _layer_s;

typedef void (*CTensor_Layer_cb)(struct _layer_s *);

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

struct _loss_s;

typedef ctensor_data_t (*CTensor_Loss_cb)(struct _loss_s *, CTensor_s *);

typedef struct _loss_s {
    // Models are really just linked lists
    // with callbacks, and hyperparameters.
    CTensor_Layer_s     *prev;
    // Forward-pass layer callback function.
    CTensor_Loss_cb     fwd;
    // Backprop-pass layer callback function.
    CTensor_Loss_cb     bckp;
    // Pointer to the 'prev' layer's 'in'.
    CTensor_s           *in;
    /*  Gradient of each 'in' element,
     *  with respect to the loss function.
     *  Gradient which will be backpropagated to
     *  the 'prev' layer. */
    CTensor_s           *in_grad;
} CTensor_Loss_s;

struct _optimizer_s;

typedef void (*CTensor_Optimize_cb)(struct _optimizer_s *, CTensor_s *, ctensor_data_t);

typedef struct _optimizer_s {
    CTensor_Optimize_cb opt;
    CTensor_Layer_cb    del;
    void                *internal;
} CTensor_Optimizer_s;

struct _model_s;

typedef struct _model_s {
    // Models are really just linked lists
    // with hyperparameters.
    // Start layer, the input layer.
    CTensor_Layer_s     *startl;
    // Last added layer, output layer.
    CTensor_Layer_s     *lastl;
    // Loss layer, not added to linked list,
    // as it will only be used during backprop.
    // lossl->prev = lastl;
    CTensor_Loss_s      *lossl;
    CTensor_Optimizer_s *optimizer;
    size_t              epochs;
    size_t              batch_size;
    size_t              batches;
    // Hyperparameters.
    ctensor_data_t      learning_rate;
} CTensor_Model_s;

/*
 *  Initialize model.
 *
 *  @param model - Struct pointer to the model.
 *  @param in_size - Number of input nodes.
*/
void ctensor_init(CTensor_Model_s *model, size_t in_size);

/*
 *  Set the next layer to the model.
 *
 *  @param model - Pointer to the model.
 *  @param out_size - Number of output nodes of the layer.
 *  @param init_cb - Init callback function
 *  (function shall be casted to CTensor_Layer_cb).
*/
CTensor_Layer_s *ctensor_add_layer(CTensor_Model_s *model, size_t out_size, CTensor_Layer_cb init_cb);

/*
 *  Define the loss function for this model.
 *
 *  @param model - Pointer to the model.
 *  @param init_cb - Init callback function
 *  (function shall be casted to CTensor_Layer_cb).
 *  
 *  @return - Loss struct layer.
*/
CTensor_Loss_s *ctensor_set_loss(CTensor_Model_s *model, CTensor_Layer_cb init_cb);

/*
 *  Define the model's gradient optimization algorithm.
 *
 *  @param model - Model to optimize.
 *  @param init_cb - Init callback function
 *  (function shall be casted to CTensor_Layer_cb).
 *
 *  @return - Optimizer layer pointer.
*/
CTensor_Optimizer_s *ctensor_set_optimizer(CTensor_Model_s *model, CTensor_Layer_cb init_cb);

/*
 *  Obtain the model's prediction, given an input.
 *
 *  @param model - Model's struct.
 *  @param input - Tensor containing the input data.
*/
CTensor_s *ctensor_predict(CTensor_Model_s *model, CTensor_s *input);

/*
 *  Test performance against an input with known outputs.
 *
 *  @param model - Model to be tested.
 *  @param input - Input tensor (must be in_sized).
 *  @param expected - Expected output tensor (must be out_sized).
*/
ctensor_data_t ctensor_test(CTensor_Model_s *model, CTensor_s *input, CTensor_s *expected);

ctensor_data_t ctensor_train(CTensor_Model_s *model, CTensor_s *x_train,
                CTensor_s *y_train, CTensor_s *x_test, CTensor_s *y_test);

/*
 *  Cleanup model, dealloc model internals.
 *
 *  @param model - Model to destroy.
*/
void ctensor_destroy(CTensor_Model_s *model);

/*
 *  Adam init function.
 *
 *  @param layer - Layer info to init.
*/
void ctensor_adam(CTensor_Optimizer_s *layer);

/*
 *  ReLU initial layer function.
 *  Fills all the layer information for the
 *  Model Abstraction API. As defined in 
 *  the documentation.
 *
 *  @param layer - Pointer of the current
 *  ReLU layer "object" to be filled.
*/
void ctensor_relu(CTensor_Layer_s *layer);

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
 *  Default initialization for FCL.
 *  Performs the Xavier-He init for the weights
 *  and a zero init for the bias.
 *  
 *  @param layer - FCL layer.
 *  @param seed - Seed for the random Xavier-He init.
*/
void ctensor_fcl_param_init(CTensor_Layer_s *layer, uint64_t seed);

/*
 *  Initializes the Loss Layer with fwd and bck
 *  callbacks.
 *
 *  @param layer - Current layer "object".
*/
void ctensor_mse_init(CTensor_Loss_s *layer);

void ctensor_ce_loss(CTensor_Loss_s *layer);

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

/*
 *  Set the whole tensor to zeros.
 *
 *  @param tensor - Tensor to be filled.
*/
void ctensor_tensor_zeros(CTensor_s *tensor);

/*
 *  Perform a multiplication between vector A and
 *  scalar alpha. Store result in vector B.
 *
 *  @param A - pointer to vector A.
 *  @param elements - Number of elements.
 *  @param alpha - Scalar alpha.
 *  @param B - pointer to vector result
*/
void ctensor_sv_mult(float *A, size_t elements, float alpha, float *B);