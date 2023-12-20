
/*
 *  Tensor implementation for CTensor.
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

#include <stddef.h>
#include <stdlib.h>

/*
 *  Allocate a new tensor.
 *
 *  @param size - Size of the new tensor.
 *
 *  @return - New allocated tensor pointer.
*/
CTensor_s *ctensor_new_tensor(size_t size)
{
    CTensor_s *tensor;

    tensor = malloc(sizeof(CTensor_s));

    // Something went wrong with allocating a new
    // tensor; we'll return NULL to indicate an issue.
    if (tensor == NULL)
        return NULL;

    // As of now, treat all tensors as 1-D Tensors.
    tensor->size = size;
    tensor->data = malloc(size * sizeof(ctensor_data_t));

    // Something went wrong with allocating data, 
    // we'll free our tensor struct, and return
    // NULL to indicate an issue.
    if (tensor->data == NULL) {
        free(tensor);
        return NULL;
    }

    return tensor;
}

/*
 *  De-allocate tensor.
 *
 *  @param tensor - Pointer to the tensor struct
 *  to be de-allocated.
*/
void *ctensor_destroy_tensor(CTensor_s *tensor)
{
    free(tensor->data);
    free(tensor);

    return;
}