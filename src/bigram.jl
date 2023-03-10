# 0 layer
struct BiGram{E}
    embed::E
end
Flux.@functor BiGram

BiGram(size::Int) = BiGram(Flux.Embedding(size, size))

function (b::BiGram)(x::Matrix{Int64})
    return b.embed(x)
end

# BiGram but with a position embedding
# perhaps this can be used to learn some context
struct BiGramWithPosition{E}
    token_embed::E
    position_embed::E
end
Flux.@functor BiGramWithPosition

BiGramWithPosition(vocabsize::Int, contextsize::Int) =
    BiGramWithPosition(Flux.Embedding(vocabsize, vocabsize), Flux.Embedding(contextsize, vocabsize))

function (b::BiGramWithPosition)(x::Matrix{Int64}, idx::Int = size(x, 1))
    tokemb = b.token_embed(x)
    posemb = b.position_embed(1:idx)
    return tokemb .+ posemb
end
