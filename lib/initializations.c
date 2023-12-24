
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
 *  @param in_size - Number of nodes in previous layer.
 *  @param out_size - Number of nodes in the current
 *  layer.
 *
 *  @return - Returns a tensor with size out_size x in_size.
*/
CTensor_s *ctensor_xavier_he_init(size_t in_size, size_t out_size, uint64_t seed)
{
    ctensor_data_t std;
    CTensor_s *tensor;
    int i;

    tensor = ctensor_randn(out_size * in_size, seed);
    if (tensor == NULL)
        return NULL;

    std = sqrtf(2.0 / in_size);

    for (i = 0; i < tensor->size; i++)
        tensor->data[i] *= std;

    return tensor;
}