module TransformerCircuits

export SelfAttention, FFN, Block, GPT, Circuit

using Flux

const A2{T} = AbstractArray{T,2} where {T<:Union{Float32,Float16}}
const A3{T} = AbstractArray{T,3} where {T<:Union{Float32,Float16}}
const A4{T} = AbstractArray{T,4} where {T<:Union{Float32,Float16}}

include("attention.jl")
include("blocks.jl")
# include("bigram.jl")
# include("ngram.jl")
include("circuit.jl")

end
