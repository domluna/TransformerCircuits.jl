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
    K = Dense(dembed, dembed, bias=false)
    V = Dense(dembed, dembed, bias=false)
    Q = Dense(dembed, dembed, bias=false)
    O = Dense(dembed, dembed, bias=false)
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
    q = attn.Q(query)
    k = attn.K(key)
    v = attn.Q(value)

    # Now we do attention for each head

    # ein"ijl,jkl->ikl"

    # nested tensor contraction
    # ein"(ijk,jkl),klm->im"(x, y, z)

    # (dembed / nheads, nheads, seqlen, batch_size)
    q = reshape(q, (attn.dhead, attn.nheads, size(q, 2), size(q, 3)))
    q = reshape(permutedims(q, (1,3,2,4)), (size(q, 1), size(q, 2),  :))
    k = reshape(k, (attn.dhead, attn.nheads, size(k, 2), size(k, 3)))
    k = permutedims(k, (1,3,2,4), size(q, 1), size(q, 2), -1)
    v = reshape(v, (attn.dhead, attn.nheads, size(v, 2), size(v, 3)))
    v = permutedims(v, (1,3,2,4))
    # (dembed / nheads, seqlen, nheads, batch_size)

    # q * k' (dembed, seqlen) * (seqlen, dembed) 
    # -> (dembed, dembed) * v (dmbed, seqlen)
    # ->  O * x (dembed, seqlen)
    # -> (dembed, seqlen)
    x = softmax((q * k') .* scale) * v

    # reshape back to (dembed) instead of (dembed ÷ nheads)
    x = attn.O(x)

    x = attn.drop(attn.O(x))
    return reshape(x, size(query))
end

(attn::SelfAttention)(inp::AbstractArray{T,N}) where {T,N} = attn(inp, inp, inp)
