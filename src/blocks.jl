function FFN(dembed::Int, dff::Int; dropout_prob = 0.1, dff_activation = gelu)
    if dff_activation == solu
        layers = Any[Dense(dembed, dff, bias = false)]
        push!(layers, solu)
        push!(layers, LayerNorm(dff))
    else
        layers = Any[Dense(dembed, dff, dff_activation, bias = false)]
    end
    push!(layers, Dense(dff, dembed, bias = false))
    push!(layers, Dropout(dropout_prob))
    Chain(layers...)
end

struct Block{A,F}
    attn::A
    ffn::F
    ln1::LayerNorm
    ln2::LayerNorm
end
Flux.@functor Block

function Block(
    dembed::Int;
    dff_activation::Function = gelu,
    nheads::Int = 1,
    dff::Int = dembed * 4,
    kwargs...,
)
    Block(
        SelfAttention(nheads, dembed),
        FFN(dembed, dff; dff_activation, kwargs...),
        LayerNorm(dembed),
        LayerNorm(dembed),
    )
end

function (b::Block)(x::AbstractArray{T}) where {T}
    x = x + b.attn(b.ln1(x))
    x = x + b.ffn(b.ln2(x))
    return x
end

struct GPT
    token_embedding::Flux.Embedding
    position_embedding::Flux.Embedding

    drop::Dropout
    encoder::Chain

    ln::LayerNorm
    decoder::Dense
end
Flux.@functor GPT

function GPT(
    blocksize::Int,
    vocabsize::Int,
    ;
    nheads::Int = 4,
    dembed::Int = 128,
    dff::Int = dembed * 4,
    dff_activation::Function = gelu,
    dropout_prob = 0.1,
    nlayers::Int = 4,
)
    vocabsize = roundto64(vocabsize)
    blocksize = roundto64(blocksize)
    GPT(
        Flux.Embedding(vocabsize, dembed),
        Flux.Embedding(blocksize, dembed),
        Dropout(dropout_prob),
        Flux.Chain([Block(dembed, dff; nheads, dropout_prob, dff_activation) for _ in 1:nlayers]...),
        LayerNorm(dembed),
        Dense(dembed, vocabsize),
    )
end

function (gpt::GPT)(x::Matrix{Int64}; idx = size(x, 1))
    tokemb = gpt.token_embed(x)
    posemb = gpt.position_embed(1:idx)
    x = tokemb .+ posemb
    x = gpt.drop(x)
    x = gpt.encoder(x)
    x = gpt.ln(x)
    return gpt.decoder(x)
end
