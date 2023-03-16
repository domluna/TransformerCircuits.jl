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
