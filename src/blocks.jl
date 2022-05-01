struct FFN
    d1::Dense
    d2::Dense
    drop::Dropout
end
Flux.@functor FFN

FFN(dembed::Int, dff::Int, dropout_prob::Float32, dff_activation) = FFN(
    Dense(dembed, dff, dff_activation, bias=false),
    Dense(dff, dembed, bias=false),
    Dropout(dropout_prob),
)

function (ffn::FFN)(x::AbstractArray{T}) where {T}
    x = ffn.d1(x)
    x = ffn.d2(x)
    ffn.drop(x)
end

struct Block{A<:AbstractAttention,F}
    attn::A
    ffn::F
    drop::Dropout
    ln1::LayerNorm
    ln2::LayerNorm
end
Flux.@functor Block

function Block(nheads::Int, dembed::Int, dff::Int, dff_activation::Function, dropout_prob::Float32)
    Block(
        SelfAttention(dembed, nheads),
        FFN(dembed, dff, dropout_prob, dff_activation),
        Dropout(dropout_prob),
    )
end
Block(dembed::Int, nheads::Int) = Block(dembed, dembed, nheads)

function Block(attn::SelfAttention, ffn::FFN, drop::Dropout)
    ln1 = LayerNorm(size(attn.O, 1))
    ln2 = LayerNorm(size(ffn.d2.W, 1))
    Block(attn, ffn, drop, ln1, ln2)
end

function (b::Block)(x::AbstractArray{T}) where {T}
    x = x + b.attn(b.ln1(x))
    x = x + b.drop(x)
    return x + b.ffn(b.ln2(x))
end

struct GPT
    position_embedding::Flux.Embedding
    token_embedding::Flux.Embedding

    drop::Dropout
    encoder::Chain

    ln::LayerNorm
    decoder::Dense

end
Flux.@functor GPT

function GPT(
    doutput::Int;
    nblocks = 1,
    nheads = 4,
    dembed = 128,
    dff = dembed * 4,
    dff_activation = gelu,
    dropout_prob::Float32 = 0.1
)
    GPT(
        Flux.Embedding(doutput, dembed),
        Flux.Embedding(doutput, dembed),
        Dropout(dropout_prob),
    Flux.Chain([
        Block(nheads, dembed, dff, dropout_prob, dff_activation)
        for _ = 1:nblocks
    ]...),
        LayerNorm(dembed),
        Dense(dembed, doutput),
    )
end

function (gpt::GPT)(x::AbstractArray{T}) where {T}
    x = gpt.embedding(x)
    x = gpt.drop(x)
    x = gpt.encoder(x)
    x = gpt.ln(x)
    return gpt.decoder(x)
end
