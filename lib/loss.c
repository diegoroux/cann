
/*
 *  Loss functions for CTensor.
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

static ctensor_data_t ctensor_mse_fwd(CTensor_Loss_s *layer, CTensor *expected);
static ctensor_data_t ctensor_mse_bckp(CTensor_Loss_s *layer, CTensor_s *expected);

/*
 *  Initializes the Loss Layer with fwd and bck
 *  callbacks.
 *
 *  @param layer - Current layer "object".
*/
void ctensor_mse_init(CTensor_Loss_s *layer)
{
    layer->fwd = (CTensor_Loss_cb)ctensor_mse_fwd;
    layer->bckp = (CTensor_Loss_cb)ctensor_mse_bckp;

    return;
}

static ctensor_data_t ctensor_mse_fwd(CTensor_Loss_s *layer, CTensor *expected)
{
    ctensor_data_t network_loss = 0.00, output_loss;
    CTensor_s *output;
    int i;

    output = layer->prev->out;

    for (i = 0; i < expected->size; i++) {
        output_loss = (expected->data[i] - output->data[i]);
        network_loss += output_loss * output_loss;
    }

    network_loss /= (ctensor_data_t)expected->size;

    return network_loss;
}

static ctensor_data_t ctensor_mse_bckp(CTensor_Loss_s *layer, CTensor_s *expected)
{
    CTensor_s *output, *in_grad;
    ctensor_data_t c;

    output = layer->prev->out;
    in_grad = layer->in_grad;

    c = 2.00/(ctensor_data_t)expected->size;

    for (i = 0; i < expected->size; i++)
        in_grad->data[i] = c * (expected->data[i] - output->data[i]);

    return 0.00;
}