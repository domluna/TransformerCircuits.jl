# self attention is unidirectional so the mask is just a lower triangular matrix
struct SelfAttention{T1,T2,D}
    K::T1
    Q::T1
    V::T1
    O::T2
    attndrop::D
    outdrop::D
    nheads::Int
end
Flux.@functor SelfAttention (K, Q, V, O, attndrop, outdrop)

function SelfAttention(nheads::Int, dembed::Int; dropout_prob = 0.1, doutput::Int = dembed)
    @assert dembed % nheads == 0 "dimension of model, $dembed must be divisible by number of heads: $nheads"
    K = Dense(dembed => dembed, bias = false)
    V = Dense(dembed => dembed, bias = false)
    Q = Dense(dembed => dembed, bias = false)
    O = Dense(dembed => doutput, bias = false)
    return SelfAttention(
        K,
        Q,
        V,
        O,
        Dropout(Float32(dropout_prob)),
        Dropout(Float32(dropout_prob)),
        nheads,
    )
end

# query, key, value dimensions are (dembed, sequence_length, batch_size)
function (sa::SelfAttention)(query::A3{T}, key::A3{T}, value::A3{T}) where {T}
    # (dembed, sequence_length, batch_size)
    q = sa.Q(query)
    k = sa.K(key)
    v = sa.V(value)
    # this takes care of reshaping the matrices to allow for per head attention
    # and then reshaping back to the original shape
    #
    # so it does
    #
    # (dembed ÷ nheads, nheads, sequence_length, batch_size)
    #
    # and then returns it back to
    #
    # (dembed, sequence_length, batch_size)
    # TODO: If the context length is known this can be static
    M = make_causal_mask(q)

    # the size of the attention matrix is (context_length, context_length, nheads, batch_size)
    x, _ = dot_product_attention(q, k, v; mask = M, nheads = sa.nheads, fdrop = sa.attndrop)
    # x, attn = dot_product_attention(q, k, v; mask = M, nheads = sa.nheads, fdrop = sa.drop)
    # @info "" size(attn) size(x) attn[:, :, 1, 1]

    x = sa.outdrop(sa.O(x))
    return x
end

(sa::SelfAttention)(inp::A3{T}) where {T} = sa(inp, inp, inp)
