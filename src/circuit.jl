struct Circuit
    token_embed::Flux.Embedding
    position_embed::Flux.Embedding
    blocks::Flux.Chain
    outtoken::Flux.Dense
end
Flux.@functor Circuit

function Circuit(vocabsize::Int, blocksize::Int, embedsize::Int; nheads::Int = 1, nlayers::Int = 2)
    Circuit(
        Flux.Embedding(vocabsize, embedsize),
        Flux.Embedding(blocksize, embedsize),
        Flux.Chain([Block(embedsize; nheads, dropout_prob = Float32(0.1)) for _ in 1:nlayers]...),
        Flux.Dense(embedsize, vocabsize),
    )
end

function (model::Circuit)(x::Matrix{Int64}; idx = size(x, 1))
    tokemb = model.token_embed(x)
    posemb = model.position_embed(1:idx)
    v = tokemb .+ posemb
    v = model.blocks(v)
    v = model.outtoken(v)
    return v
end
(model::Circuit)(x::Vector{Int64}; kwargs...) = model(reshape(x, :, 1); kwargs...)
