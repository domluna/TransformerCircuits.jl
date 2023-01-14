struct FFN
    embed::Dense
    proj::Dense
    drop::Dropout
end
Flux.@functor FFN

function FFN(dembed::Int, dff::Int; dropout_prob = 0.1, dff_activation = gelu)
    FFN(
        Dense(dembed, dff, dff_activation, bias = false),
        Dense(dff, dembed, bias = false),
        Dropout(dropout_prob),
    )
end

function (ffn::FFN)(x::AbstractArray{T}) where {T}
    x = ffn.embed(x)
    x = ffn.proj(x)
    ffn.drop(x)
end

struct Block{A,F}
    attn::A
    ffn::F
    ln1::LayerNorm
    ln2::LayerNorm
end
Flux.@functor Block

function Block(
    dembed::Int,
    dff::Int;
    dff_activation::Function = gelu,
    nheads::Int = 1,
    kwargs...,
)
    Block(SelfAttention(nheads, dembed), FFN(dembed, dff; dff_activation, kwargs...))
end

function Block(attn::SelfAttention, ffn::FFN)
    ln1 = LayerNorm(size(attn.O.weight, 1))
    ln2 = LayerNorm(size(ffn.proj.weight, 1))
    Block(attn, ffn, ln1, ln2)
end

function (b::Block)(x::AbstractArray{T}) where {T}
    x = x + b.attn(b.ln1(x))
    x = x + b.ffn(b.ln2(x))
    return x
end

struct GPT
    token_embedding::Flux.Embedding

    drop::Dropout
    encoder::Chain

    ln::LayerNorm
    decoder::Dense
end
Flux.@functor GPT

function GPT(;
    vocab_size::Int = 50257,
    nheads::Int = 4,
    dembed::Int = 128,
    dff::Int = dembed * 4,
    dff_activation::Function = gelu,
    dropout_prob = 0.1,
    nlayers::Int = 4,
)
    GPT(
        Flux.Embedding(vocab_size, dembed),
        Dropout(dropout_prob),
        Flux.Chain(
            [
                Block(dembed, dff; nheads, dropout_prob, dff_activation) for _ in 1:nlayers
            ]...,
        ),
        LayerNorm(dembed),
        Dense(dembed, vocab_size),
    )
end

function (gpt::GPT)(x::AbstractArray{T}) where {T}
    x = gpt.token_embedding(x)
    x = gpt.drop(x)
    x = gpt.encoder(x)
    x = gpt.ln(x)
    return gpt.decoder(x)
end
