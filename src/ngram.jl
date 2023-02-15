struct NGram
    token_embed::Flux.Embedding
    position_embed::Flux.Embedding
    block::Block
    outtoken::Flux.Dense
end
Flux.@functor NGram

function NGram(vocab_size::Int, block_size::Int, embed_size::Int; nheads::Int = 1)
    NGram(
        Flux.Embedding(vocab_size, embed_size),
        Flux.Embedding(block_size, embed_size),
        Block(embed_size; nheads = nheads),
        Flux.Dense(embed_size, vocab_size),
    )
end

function (model::NGram)(x::Matrix{Int64})
    tokemb = model.token_embed(x)
    T = size(model.position_embed.weight, 2)
    posemb = model.position_embed(1:T)
    v = tokemb .+ posemb
    v = model.block(v)
    v = model.outtoken(v)
    return softmax(v, dims = 1)
end
