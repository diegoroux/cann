
/*
 *  Adam Stochastic Optimization for CTensor.
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
 *  Adam Stochastic Optimization Algorithm
 *  
 *  @param grad - evaluated gradient vector.
 *  @param grad_size - number of elements in gradient vector.
 *  @param m - first moment vector.
 *  @param v - second moment vector.
 *  @param b1 - first bias correction.
 *  @param b2 - second bias correction.
 *  @param lr - learning rate.
 *  @param t - Number of gradient update.
*/
void _ct_adam(float *grad, size_t grad_size, float *m, float *v, float b1, float b2, float lr, int t)
{
    float mb, vb;
    int i;

    for (i = 0; i < grad_size; i++) {
        m[i] = b1 * m[i] + (1 - b1) * grad[i];
        v[i] = b2 * v[i] + (1 - b2) * (grad[i] * grad[i]);

        mb = m[i] / (1 - powf(b1, t));
        vb = v[i] / (1 - powf(b2, t));

        grad[i] = (-lr * mb) / (sqrtf(vb) + 1e-7);
    }

    return;
}