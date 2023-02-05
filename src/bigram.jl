struct Bigram
    embed::Flux.Embedding
end
Flux.@functor Bigram

Bigram(size::Int) = Bigram(Flux.Embedding(size, size))

function (b::Bigram)(x::Matrix{Int64})
    x = b.embed(x)
    x = softmax(x, dims = 1)
    return x
end
