abstract type AbstractAttention end

struct SelfAttention <: AbstractAttention
    K::Dense
    Q::Dense
    V::Dense
    O::Dense

    drop::Dropout

    # mask::Matrix

    dhead::Int
    nheads::Int
end
Flux.@functor SelfAttention (K, Q, V, O)

function SelfAttention(dembed::Int, nheads::Int, dropout_prob::Float32)
    @assert dembed % nheads != 0 "dimension of model, $dembed must be divisible by number of heads: $nheads"
    dhead = dembed ÷ nheads
    K = Dense(dembed, dhead * nheads)
    V = Dense(dembed, dhead * nheads)
    Q = Dense(dembed, dhead * nheads)
    O = Dense(dembed, dhead * nheads)
    return SelfAttention(K, Q, V, O, Dropout(dropout_prob), dhead, nheads)
end


# query, key, value dimensions are (dembed, seqlen, batch_size)
# query, key, value dimensions are (dhead, nheads, seqlen, batch_size)
function (attn::SelfAttention)(
    query::AbstractArray{T,N},
    key::AbstractArray{T,N},
    value::AbstractArray{T,N},
) where {T,N}
    scale = T(1 / sqrt(attn.dhead))
    Q = attn.Q(reshape(query, size(query, 1), :))
    K = attn.K(reshape(key, size(key, 1), :))
    V = attn.V(reshape(value, size(value, 1), :))
    #  Q (features, seq_len, batch_size)
    #  K (batch_size, seq_len, features)
    x = softmax((Q * K') .* scale) * V
    x = attn.drop(attn.O(x))
    return reshape(x, size(query))
end

function (attn::SelfAttention)(inp)
    return attn(inp, inp, inp)
end

function (attn::SelfAttention)(
    encoder_output,
    inp,
)
    return attn(encoder_output, encoder_output, inp)
end
