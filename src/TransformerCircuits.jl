module TransformerCircuits

export SelfAttention, FFN, Block, GPT

using Flux
using NeuralAttentionlib

const A3{T} = AbstractArray{T, 3}

include("encoding.jl")
include("attention.jl")
include("blocks.jl")

end
