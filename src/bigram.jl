# 0 layer
struct BiGram{E}
    embed::E
end
Flux.@functor BiGram

BiGram(size::Int) = BiGram(Flux.Embedding(size, size))

function (b::BiGram)(x::Matrix{Int64})
    x = b.embed(x)
    x = softmax(x, dims = 1)
    return x
end
(b::BiGram)(x::Vector{Int64}) = b(reshape(x, :, 1))
