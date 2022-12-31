# self attention is unidirectional so the mask is just a lower triangular matrix
struct SelfAttention
    K::Dense
    Q::Dense
    V::Dense
    O::Dense

    drop::Dropout

    dhead::Int
    nheads::Int
end
Flux.@functor SelfAttention (K, Q, V, O)

function SelfAttention(dembed::Int, nheads::Int; dropout_prob=0.0)
    @assert dembed % nheads == 0 "dimension of model, $dembed must be divisible by number of heads: $nheads"
    dhead = dembed ÷ nheads
    K = Dense(dembed => dembed, bias=false)
    V = Dense(dembed => dembed, bias=false)
    Q = Dense(dembed => dembed, bias=false)
    O = Dense(dembed => dembed, bias=false)
    return SelfAttention(K, Q, V, O, Dropout(dropout_prob), dhead, nheads)
end


# query, key, value dimensions are (dembed, sequence_length, batch_size)
function (sa::SelfAttention)(
    query::A3{T},
    key::A3{T},
    value::A3{T},
) where {T}
    # (dembed, sequence_length, batch_size)
    q = sa.Q(query)
    k = sa.K(key)
    v = sa.Q(value)
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
    x = NeuralAttentionlib.multihead_qkv_attention(sa.nheads, q, k, v)

    x = sa.drop(sa.O(x))
    return x
end

(sa::SelfAttention)(inp::A3{T}) where {T} = sa(inp, inp, inp)
