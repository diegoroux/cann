
/*
 *  Model Abstraction API for CTensor.
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

void ctensor_init(CTensor_Model_s *model, size_t in_size)
{
    CTensor_Layer_s *in_layer;

    in_layer = (CTensor_Layer_s *)malloc(sizeof(CTensor_Layer_s));

    model->startl = in_layer;
    model->lastl = in_layer;

    // Initialize layer.
    in_layer->prev = NULL;
    in_layer->next = NULL;
    in_layer->in = NULL;
    // Instead of allocating a new tensor, and copying inputs to it
    // we're just going to keep replacing the pointer to the
    // data input.
    in_layer->out = (CTensor_s *)malloc(sizeof(CTensor_s));
    in_layer->out->data = NULL;
    in_layer->out->size = in_size;
    // Our input layer has no callbacks.
    in_layer->fwd = NULL;
    in_layer->bckp = NULL;
    in_layer->update = NULL;
    in_layer->del = NULL;
    // In order for any layer to work, according to the API,
    // it needs to produce a gradient with respect to its
    // inputs.
    in_layer->in_grad = ctensor_new_tensor(in_size);
    in_layer->loss_grad = NULL;
    // The start layer has no internal/learnable parameters.
    in_layer->internal_grad = NULL;

    return;
}

CTensor_Layer_s *ctensor_add_layer(CTensor_Model_s *model, size_t out_size, CTensor_Layer_cb init_cb)
{
    CTensor_Layer_s *layer, *pos;

    layer = (CTensor_Layer_s *)malloc(sizeof(CTensor_Layer_s));

    // Get the last layer in the linked list.
    pos = model->lastl;
    // Set our layer as the last.
    model->lastl = layer;

    // Add 'layer' to the linked list.
    pos->next = layer;

    // Initialize layer.
    layer->next = NULL;
    layer->prev = pos;
    layer->in = pos->out;
    // Allocate tensors, according to the given size.
    layer->out = ctensor_new_tensor(out_size);
    layer->in_grad = ctensor_new_tensor(layer->in->size);

    // The 'loss gradient' is the gradient that comes
    // from the next layer. We exploit chain rule
    // to just calculate a local gradient and then
    // multiply by the 'loss gradient'. 
    pos->loss_grad = layer->in_grad;

    // Initialize layer internals (if any), and get all its
    // callbacks.
    init_cb(layer);

    // Layers can have configurable parameters, that are
    // independent from us, so we return the pointer to 
    // this layer in order for the user to be able 
    // to perform any config.
    return layer;
}

/*
 *  Define the loss function for this model.
 *
 *  @param model - Pointer to the model.
 *  @param init_cb - Init callback function
 *  (function shall be casted to CTensor_Layer_cb).
 *  
 *  @return - Loss struct layer.
*/
CTensor_Loss_s *ctensor_set_loss(CTensor_Model_s *model, CTensor_Layer_cb init_cb)
{
    CTensor_Loss_s *loss;

    loss = (CTensor_Loss_s *)malloc(sizeof(CTensor_Loss_s));

    model->lossl = loss;

    loss->prev = model->lastl;
    loss->in = loss->prev->out;
    loss->in_grad = ctensor_new_tensor(loss->in->size);

    model->lastl->loss_grad = loss->in_grad;

    init_cb((void *)loss);

    return loss;
}

/*
 *  Define the model's gradient optimization algorithm.
 *
 *  @param model - Model to optimize.
 *  @param init_cb - Init callback function
 *  (function shall be casted to CTensor_Layer_cb).
 *
 *  @return - Optimizer layer pointer.
*/
CTensor_Optimizer_s *ctensor_set_optimizer(CTensor_Model_s *model, CTensor_Layer_cb init_cb)
{
    CTensor_Optimizer_s *opt;

    opt = (CTensor_Optimizer_s *)malloc(sizeof(CTensor_Optimizer_s));
    model->optimizer = opt;

    init_cb((void *)opt);

    return opt;
}

/*
 *  Obtain the model's prediction, given an input.
 *
 *  @param model - Model's struct.
 *  @param input - Tensor containing the input data.
*/
CTensor_s *ctensor_predict(CTensor_Model_s *model, CTensor_s *input)
{
    CTensor_Layer_s *pos;

    // Get the Input Layer.
    pos = model->startl;
    // Point the Input Layer to the input's data pointer.
    pos->out->data = input->data;

    // Do the forward pass.
    pos = pos->next;

    while (pos != NULL) {
        pos->fwd(pos);
        pos = pos->next;
    }

    // Get the last layer.
    pos = model->lastl;

    return pos->out;
}

ctensor_data_t ctensor_test(CTensor_Model_s *model, CTensor_s *input, CTensor_s *expected)
{
    CTensor_Loss_s *lossl;
    CTensor_Layer_s *pos;
    ctensor_data_t loss;

    pos = model->startl;
    pos->out->data = input->data;

    pos = pos->next;

    while (pos != NULL) {
        pos->fwd(pos);
        pos = pos->next;
    }

    lossl = model->lossl;
    loss = lossl->fwd(lossl, expected);

    return loss;
}

static inline void _ct_do_bckp(CTensor_Model_s *model, CTensor_s *grad)
{
    CTensor_Layer_s *pos;
    size_t grad_size;

    grad_size = grad->size;

    pos = model->lastl;

    while (pos != NULL) {
        if (pos->bckp == NULL) 
            break;

        pos->bckp(pos);

        if (pos->internal_grad == NULL) {
            pos = pos->prev;
            continue;
        }

        ctensor_vector_sum(pos->internal_grad->data, pos->internal_grad->size,
                    grad->data, grad->data);

        grad->data += pos->internal_grad->size;

        pos = pos->prev;
    }

    grad->data -= grad_size;

    return;
}

static inline void _ct_grad_update(CTensor_Model_s *model, CTensor_s *grad)
{
    CTensor_Layer_s *pos;
    size_t grad_size;
    int i;

    grad_size = grad->size;

    pos = model->lastl;

    while (pos != NULL) {
        if (pos->internal_grad == NULL) {
            pos = pos->prev;
            continue;
        }

        for (i = 0; i < pos->internal_grad->size; i++) {
            pos->internal_grad->data[i] = grad->data[i];
        }

        grad->data += pos->internal_grad->size;

        pos->update(pos);
        pos = pos->prev;
    }

    grad->data -= grad_size;

    return;
}

static inline size_t _ct_get_model_param_size(CTensor_Model_s *model)
{
    CTensor_Layer_s *pos;
    size_t size = 0;

    pos = model->startl->next;

    while (pos != NULL) {
        if (pos->internal_grad != NULL)
            size += pos->internal_grad->size;

        pos = pos->next;
    }

    return size;
}

static inline ctensor_data_t _ct_train_batch(CTensor_Model_s *model,
                    CTensor_s *x_train, CTensor_s *y_train, CTensor_s *x_test,
                    CTensor_s *y_test, CTensor_s *avg_grad)
{
    ctensor_data_t loss, vloss, batch_loss = 0.00, avg;
    CTensor_Loss_s *lossl;
    size_t out_s, in_s;
    int i;

    out_s = model->lastl->out->size;
    in_s = model->startl->out->size;

    // Get the model's loss layer.
    lossl = model->lossl;

    // Clear our average gradient tensor.
    ctensor_tensor_zeros(avg_grad);

    for (i = 0; i < model->batch_size; i++) {
        // Check how we're doing, by getting the loss
        // with respect to the test set, data that our network
        // hasn't been trained on.
        vloss = ctensor_test(model, x_test, y_test);
        // Get the loss with respect to the training set.
        loss = ctensor_test(model, x_train, y_train);

        // TODO: Check how much vloss is changing to prevent overfitting.

        // Add out loss to the overall batch loss (we'll average it
        // later on).
        batch_loss += loss;

        // Do the backprop on the loss function, to start the chain rule.
        lossl->bckp(lossl, y_train);
        // Walk through all of our model and perform backprop.
        _ct_do_bckp(model, avg_grad);

        // Batches are stored contiguosly in memory.
        // So we just jump to the next example in our batch.
        x_train->data += in_s;
        y_train->data += out_s;
    }

    // Reset the pointers.
    x_train->data -= x_train->size;
    y_train->data -= y_train->size;

    avg = 1.00/(ctensor_data_t)model->batch_size;

    // Average our loss.
    batch_loss *= avg;
    // Average all gradients (we're doing a mini-batch update).
    ctensor_sv_mult(avg_grad->data, avg_grad->size, avg, avg_grad->data);

    // Run the average gradients through the selected 
    // optimizer gradient function.
    model->optimizer->opt((void *)model->optimizer,
                            avg_grad, model->learning_rate);
    // Perform the update on the model's parameters.
    _ct_grad_update(model, avg_grad);

    return batch_loss;
}

ctensor_data_t ctensor_train(CTensor_Model_s *model, CTensor_s *x_train,
                CTensor_s *y_train, CTensor_s *x_test, CTensor_s *y_test)
{
    ctensor_data_t network_loss;
    CTensor_s *avg_grad = NULL;
    size_t grad_size = 0;
    int epoch, batch;

    grad_size = _ct_get_model_param_size(model);
    avg_grad = ctensor_new_tensor(grad_size);

    for (epoch = 0; epoch < model->epochs; epoch++) {
        network_loss = 0.00;

        for (batch = 0; batch < model->batches; batch++)
            network_loss += _ct_train_batch(model, x_train, y_train, x_test, y_test, avg_grad);

        network_loss /= model->batches;
    }

    ctensor_destroy_tensor(avg_grad);

    return network_loss;
}

void ctensor_destroy(CTensor_Model_s *model)
{
    CTensor_Layer_s *pos, *prev;
    CTensor_Optimizer_s *opt;
    CTensor_Loss_s *loss;

    pos = model->lastl;

    while (pos != NULL) {
        if (pos->del != NULL)
            pos->del(pos);

        ctensor_destroy_tensor(pos->in_grad);

        if (pos->prev == NULL) {
            free(pos->out);
        } else {
            ctensor_destroy_tensor(pos->out);
        }

        prev = pos->prev;
        free(pos);

        pos = prev;
    }

    loss = model->lossl;

    if (loss != NULL) {
        ctensor_destroy_tensor(loss->in_grad);
        free(loss);
    }

    opt = model->optimizer;

    if (opt != NULL) {
        opt->del((void *)opt);
        free(opt);
    }

    return;
}