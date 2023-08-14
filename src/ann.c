
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

#include "ann_internal.h"

void ann_init(ann_ctx *ctx, size_t input_nodes,
            size_t hidden_layers, size_t hidden_nodes,
            size_t output_nodes)
{
    if (ctx == NULL)
        return;

    ctx->no_input_nodes = input_nodes;
    ctx->no_hidden_layers = hidden_layers;
    ctx->no_hidden_nodes = hidden_nodes;
    ctx->no_output_nodes = output_nodes;

    mempool_init(&ctx->mempool,
                ANN_SIZE(input_nodes, hidden_layers,
                hidden_nodes, output_nodes));

    if (ctx->mempool.mem == NULL)
        return;
    
    ctx->weight_matrix = mempool_alloc(&ctx->mempool,
                        (1 + hidden_layers) * (sizeof(matrix *)));

    if (ctx->weight_matrix == NULL)
        return;

    return;
}

void ann_free(ann_ctx *ctx)
{
    if (ctx == NULL)
        return;

    ctx->no_input_nodes = 0;
    ctx->no_hidden_layers = 0;
    ctx->no_hidden_nodes = 0;
    ctx->no_output_nodes = 0;

    mempool_free(&ctx->mempool);
    
    return;
}