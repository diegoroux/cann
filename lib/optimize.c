
/*
 *  Stochastic Optimizers for CTensor.
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

void _ct_adam(float *grad, size_t grad_size, float *m, float *v, float b1, float b2, float lr, int t);
void _ct_adam_opt(CTensor_Optimizer_s *layer, CTensor_s *grad, ctensor_data_t learning_rate);
void _ct_adam_destroy(CTensor_Optimizer_s *layer);

typedef struct {
    int             t;
    ctensor_data_t  b1;
    ctensor_data_t  b2;
    CTensor_s       *m;
    CTensor_s       *v;
} _adam_s;

/*
 *  Adam init function.
 *
 *  @param model - Current model to optimize.
*/
void ctensor_adam(CTensor_Optimizer_s *layer)
{
    _adam_s *data;

    layer->opt = _ct_adam_opt;
    layer->del = (CTensor_Layer_cb)_ct_adam_destroy;
    layer->internal = malloc(sizeof(_adam_s));

    data = (_adam_s *)layer->internal;

    data->t = 1;
    data->b1 = 0.99;
    data->b2 = 0.999;
    data->m = NULL;
    data->v = NULL;

    return;
}

void _ct_adam_opt(CTensor_Optimizer_s *layer, CTensor_s *grad, ctensor_data_t learning_rate)
{
    _adam_s *data;
    int t;

    data = (_adam_s *)layer->internal;

    if (data->m == NULL) {
        data->m = ctensor_new_tensor(grad->size);
        ctensor_tensor_zeros(data->m);
    }

    if (data->v == NULL) {
        data->v = ctensor_new_tensor(grad->size);
        ctensor_tensor_zeros(data->v);
    }

    t = data->t++;

    _ct_adam(grad->data, grad->size, data->m->data, data->v->data,
            data->b1, data->b2, learning_rate, t);

    return;
}

void _ct_adam_destroy(CTensor_Optimizer_s *layer)
{
    _adam_s *data;

    data = (_adam_s *)layer->internal;

    ctensor_destroy_tensor(data->m);
    ctensor_destroy_tensor(data->v);

    free(data);

    return;
}