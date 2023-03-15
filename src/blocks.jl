struct FFN{D1,D2,D}
    dense1::D1
    dense2::D2
    drop::D
end
Flux.@functor FFN

function FFN(dembed::Int, dff::Int; dropout_prob = 0.1, dff_activation = gelu)
    FFN(
        Dense(dembed, dff, dff_activation, bias = false),
        Dense(dff, dembed, bias = false),
        Dropout(Float32(dropout_prob)),
    )
end

function (f::FFN)(x::A3{T}) where {T}
    # o = Array{T}(undef, size(x))
    return f.drop(f.dense2(f.dense1(x)))
    return x
end

struct Block{A,F,L}
    attn::A
    ffn::F
    ln1::L
    ln2::L
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

function (b::Block)(x::A3{T}) where {T}
    # o = Array{T}(undef, size(x))
    x = x + b.attn(b.ln1(x))
    x = x + b.ffn(b.ln2(x))
    return x
end

struct GPT{E,D,B,L,O}
    token_embedding::E
    position_embedding::E

    drop::D
    encoder::B

    ln::L
    decoder::O
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
    GPT(
        Flux.Embedding(vocabsize, dembed),
        Flux.Embedding(blocksize, dembed),
        Dropout(Float32(dropout_prob)),
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
