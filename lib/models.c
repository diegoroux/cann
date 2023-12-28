
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

    in_layer->prev = NULL;
    in_layer->next = NULL;
    in_layer->in = NULL;
    in_layer->out = (CTensor_s *)malloc(sizeof(CTensor_s));
    in_layer->out->size = in_size;
    in_layer->in_grad = ctensor_new_tensor(in_size);
    in_layer->loss_grad = NULL;

    return;
}

CTensor_Layer_s *ctensor_add_layer(CTensor_Model_s *model, size_t out_size, CTensor_Layer_cb init_cb)
{
    CTensor_Layer_s *layer, *pos;

    layer = (CTensor_Layer_s *)malloc(sizeof(CTensor_Layer_s));

    pos = model->lastl;
    model->lastl = layer;

    pos->next = layer;

    layer->next = NULL;
    layer->prev = pos;
    layer->in = pos->out;
    layer->out = ctensor_new_tensor(out_size);
    layer->in_grad = ctensor_new_tensor(layer->in->size);
    layer->loss_grad = pos->in_grad;

    init_cb(layer);

    return layer;
}

CTensor_Loss_s *ctensor_set_loss(CTensor_Model_s *model, CTensor_Layer_cb init_cb)
{
    CTensor_Loss_s *loss;

    loss = (CTensor_Loss_s *)malloc(sizeof(CTensor_Loss_s));

    model->lossl = loss;

    loss->prev = model->lastl;
    loss->in = loss->prev->out;
    loss->in_grad = ctensor_new_tensor(loss->in->size);

    init_cb((void *)loss);

    return loss;
}

CTensor_s *ctensor_predict(CTensor_Model_s *model, CTensor_s *input)
{
    CTensor_Layer_s *pos;

    pos = model->startl;
    pos->out->data = input->data;

    pos = pos->next;

    while (pos != NULL) {
        pos->fwd(pos);
        pos = pos->next;
    }

    pos = model->lastl;

    return pos->out;
}

void ctensor_destroy(CTensor_Model_s *model)
{
    CTensor_Layer_s *pos, *prev;
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

    return;
}