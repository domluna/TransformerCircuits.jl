struct Circuit
    token_embed::Flux.Embedding
    position_embed::Flux.Embedding
    blocks::Flux.Chain
    outtoken::Flux.Dense
end
Flux.@functor Circuit

function Circuit(
    vocab_size::Int,
    block_size::Int,
    embed_size::Int;
    nheads::Int = 1,
    nlayers::Int = 2,
)
    Circuit(
        Flux.Embedding(vocab_size, embed_size),
        Flux.Embedding(block_size, embed_size),
        Flux.Chain(
            [
                Block(
                    embed_size;
                    nheads,
                    dropout_prob = Float32(0.1),
                ) for _ = 1:nlayers
            ]...,
        ),
        Flux.Dense(embed_size, vocab_size),
    )
end

function (model::Circuit)(x::Matrix{Int64})
    tokemb = model.token_embed(x)
    T = size(model.position_embed.weight, 2)
    posemb = model.position_embed(1:T)
    v = tokemb .+ posemb
    v = model.blocks(v)
    v = model.outtoken(v)
    return softmax(v, dims = 1)
end
