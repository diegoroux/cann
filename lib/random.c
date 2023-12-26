
/*
 *  RNGs (Random Number Generators) for CTensor.
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

#include <stdint.h>
#include <math.h>

#define ROTL(x, r) ((x << r) | (x >> (32 - r)))

/*
 *  SplitMax64 implementation by Sebastiano Vigna.
 *
 *  Taken from https://prng.di.unimi.it/
 *
 *  Adapted to 'return' two uint32_t numbers.
*/
static void splitmix64(uint64_t *x, uint32_t *res) {
    uint64_t z;

    z = (*x ^ (*x >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z ^= (z >> 31);

    *x = *x + 0x9e3779b97f4a7c15;

    // Extract the upper 32-bits.
    res[0] = (z >> 32);

    // Extract the lower 32-bits.
    res[1] = (z & 0xffffffff); 

    return;
}

/*
 *  xoshiro128+ implementation by David Blackman
 *  and Sebastiano Vigna.
 *
 *  Taken from https://prng.di.unimi.it/
 *
 *  Adapted to return uniform floats 
 *  in the range of [0, 1).
*/
static float xoshiro128p(uint32_t *s) {
    uint32_t res, t;
    float *res_f;

    // Obtain the upper 23 bits, and 'normalize'
    // the float by setting it's exponent to 127 (0).
    res = ((s[0] + s[3]) >> 9) | UINT32_C(0x3f800000);

    // 'Cast' it to a float. 
    res_f = (float *)&res;

    // xoshiro128+ algorithm.
    t = s[1] << 9;

    s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = ROTL(s[3], 11);

    // res_f will be a uniformlly distributed float in the range
    // [1, 2), by substracting 1.0, we're shifting the range to
    // [0, 1).
    return *res_f - 1.0;
}

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
void ctensor_randu(CTensor_s *tensor, uint64_t seed)
{
    uint32_t state[4];
    int i;

    // Fill the 128-bit state with randomness.
    splitmix64(&seed, state);
    splitmix64(&seed, &state[2]);

    // Generate all numbers.
    for (i = 0; i < tensor->size; i++)
        tensor->data[i] = xoshiro128p(state);

    return;
}

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
void ctensor_randn(CTensor_s *tensor, uint64_t seed)
{
    uint32_t state[4];
    float x, y, s;
    int i;

    // Fill the 128-bit state with randomness.
    splitmix64(&seed, state);
    splitmix64(&seed, &state[2]);

    // Generate all the random numbers.
    for (i = 0; i < tensor->size; i++) {
        // We'll be generating two floats per instance,
        // so all 'even' iterations will generate 2 floats
        // and all 'odd' iterations will use the remaning
        // unused float.
        switch (i % 2) {
            case 0:
                // Let x and y, be two uniformly distributed random
                // numbers in the range [-1, 1).
                //
                // (Note that xoshiro128p returns uniformly distributed
                // numbers in the range of [0, 1), so multiplying by 2.0
                // we shift the range to [0, 2) and by subtracting 
                // 1.0 we now shift this range to [-1, 1).)
                //
                // The sum of their squares (defined as s) shall be
                // in the range of (0, 1), if not, x and y shall be
                // discarded and new x and y be chosen.
                do {
                    x = xoshiro128p(state) * 2.0 - 1.0;
                    y = xoshiro128p(state) * 2.0 - 1.0;
                    s = x*x + y*y;
                } while ((s == 0) || (s >= 1));

                // The chi variate is defined as follows:
                // sqrt(-2 * log(s))
                // The full formula to generate one of the pairs is
                // x/sqrt(s) * chi, where x/sqrt(s), y/sqrt(s)
                // represent either the cosine or sine of the angle
                // of the vector (x, y).
                // Because dividing by sqrt(s) is a commonality 
                // between them, we include it in the chi variate
                // calculation.
                s = sqrtf(-2.0 * logf(s) / s);

                // We obtain the first float of the pair as defined
                // x * chi, remembering that 1/sqrt(s) is now
                // included in chi.
                tensor->data[i] = x * s;
                break;
            case 1:
                // We reuse the generated chi value, and obtain the
                // second of the pair. 
                tensor->data[i] = y * s;
                break;
        }
    }

    return;
}