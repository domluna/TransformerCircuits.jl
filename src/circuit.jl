struct Circuit{E,B,O}
    token_embed::E
    position_embed::E
    blocks::B
    outtoken::O
end
Flux.@functor Circuit

function Circuit(
    vocabsize::Int,
    blocksize::Int,
    embedsize::Int;
    nheads::Int = 1,
    nlayers::Int = 2,
    dropout_prob::Float32 = Float32(0.1),
)
    Circuit(
        Flux.Embedding(vocabsize, embedsize),
        Flux.Embedding(blocksize, embedsize),
        Flux.Chain([Block(embedsize; nheads, dropout_prob) for _ in 1:nlayers]...),
        Flux.Dense(embedsize, vocabsize),
    )
end

function (model::Circuit)(x::AbstractMatrix{Int64}; idx = size(x, 1))
    tokemb = model.token_embed(x)
    posemb = model.position_embed(1:idx)
    o = tokemb .+ posemb
    o = model.blocks(o)
    o = model.outtoken(o)
    return o
end
