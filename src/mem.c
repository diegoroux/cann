
/*
 *  Copyright (C) 2023 Diego Roux
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, version 3 of the License.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include "ann_internal.h"

// TODO: Create an alternative for embed without malloc.
void mempool_init(mempool_ctx *ctx, size_t size)
{
    if (ctx == NULL || size == 0)
        return;

    ctx->mem = malloc(size);

    if (ctx->mem == NULL)
        return;

    ctx->size = size;
    ctx->used = 0;

    return;
}

void *mempool_alloc(mempool_ctx *ctx, size_t size)
{
    if (ctx == NULL || size == 0)
        return NULL;

    if (ctx->mem == NULL || ctx->size == 0)
        return NULL;

    if ((ctx->size - ctx->used) >= size) {
        ctx->used += size;
        return ctx->mem + size;
    }

    return NULL;
}

void mempool_free(mempool_ctx *ctx)
{
    if (ctx != NULL) {
        if (ctx->mem != NULL) {
            free(ctx->mem);
            ctx->size = 0;
            ctx->used = 0;
        }
    }

    return;
}