struct Embedding
    W::Matrix
end

Flux.@functor Embedding (W,)

Embedding(embeddings::Int, dembed::Int; init=Flux.glorot_normal) = Embedding(init(dembed, embeddings))

(e::Embedding)(x::Vector{T}) where T = e.W * Flux.onehotbatch(x, 1:size(e.W, 2))
(e::Embedding)(x::Array{T}) where T = e(reshape(x, :))

