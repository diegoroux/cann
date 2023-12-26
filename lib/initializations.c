
/*
 *  Initialization functions for CTensor.
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

#include <math.h>

/*
 *  Initialize weights using the Xavier-He initialization
 *  method.
 *
 *  @param tensor - Pointer to the Tensor to be initialization.
 *  @param in_size - Number of nodes in the previous layer.
 *  @param seed - Seed for the PRNG.
*/
void ctensor_xavier_he_init(CTensor_s *tensor, size_t in_size, uint64_t seed)
{
    ctensor_data_t std;
    int i;

    // Fill the tensor with random numbers sampled from a normal distribution.
    ctensor_randn(tensor, seed);

    std = sqrtf(2.0 / in_size);

    for (i = 0; i < tensor->size; i++)
        tensor->data[i] *= std;

    return tensor;
}

/*
 *  Initialize weights using the Xavier initialization
 *  method.
 *
 *  @param tensor - Pointer to the Tensor to be initialization.
 *  @param in_size - Number of nodes in the previous layer.
 *  @param seed - Seed for the PRNG.
*/
void ctensor_xavier_init(CTensor_s *tensor, size_t in_size, uint64_t seed)
{
    ctensor_data_t std;
    int i;

    // Fill the tensor with random numbers sampled from a normal distribution.
    ctensor_randn(tensor, seed);

    std = sqrtf(1.00 / in_size);

    for (i = 0; i < tensor->size; i++)
        tensor->data[i] *= std;

    return tensor;
}
