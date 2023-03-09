module TransformerCircuits

export SelfAttention, FFN, Block, GPT, Circuit

using Flux

const A3{T} = Array{T,3} where {T<:Union{Float32,Float16}}

# https://transformer-circuits.pub/2022/solu
solu(x) = x .* softmax(x)
roundto64(x) = x % 64 != 0 ? x + 64 - x % 64 : x

include("attention.jl")
include("blocks.jl")
# include("bigram.jl")
# include("ngram.jl")
include("circuit.jl")

end
