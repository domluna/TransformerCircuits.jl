abstract type AbstractAttention end

struct SelfAttention <: AbstractAttention
    K::Dense
    Q::Dense
    V::Dense
    O::Dense

    drop::Dropout

    mask::Matrix

    dhead::Int
    nheads::Int
end
Flux.@functor SelfAttention (K, Q, V, O)

function SelfAttention(dembed::Int, nheads::Int, dropout_prob::Float32)
    @assert dembed % nheads != 0 "dimension of model, $dembed must be divisible by number of heads: $nheads"
    dhead = dembed ÷ nheads
    K = Dense(dembed, dhead * nheads, bias=false)
    V = Dense(dembed, dhead * nheads, bias=false)
    Q = Dense(dembed, dhead * nheads, bias=false)
    O = Dense(dembed, dhead * nheads, bias=false)
    return SelfAttention(K, Q, V, O, Dropout(dropout_prob), dhead, nheads)
end


# query, key, value dimensions are (dembed, seqlen, batch_size)
function (attn::SelfAttention)(
    query::AbstractArray{T,N},
    key::AbstractArray{T,N},
    value::AbstractArray{T,N},
) where {T,N}
    scale = T(1 / sqrt(attn.dhead))

    # (dembed, seqlen, batch_size)
    Q = attn.Q(query)
    K = attn.K(key)
    V = attn.Q(value)

    # Now we do attention for each head

    # (dembed / nheads, nheads, seqlen, batch_size)
    q = reshape(Q, (attn.dhead, attn.nheads, size(Q, 2), size(Q, 3)))
    q2 = reshape(permutedims(q, (1,3,2,4)), (size(q, 1), size(q, 2),  :))
    k = reshape(K, (attn.dhead, attn.nheads, size(K, 2), size(K, 3)))
    k2 = permutedims(k, (1,3,2,4), size(q, 1), size(q, 2), -1)
    v = reshape(V, (attn.dhead, attn.nheads, size(V, 2), size(V, 3)))
    v2 = permutedims(v, (1,3,2,4))
    # (dembed / nheads, seqlen, nheads, batch_size)

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
