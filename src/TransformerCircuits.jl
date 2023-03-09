module TransformerCircuits

export Circuit, BiGram

using Flux

const A3{T} = Array{T,3} where {T<:Union{Float32,Float16}}

# https://transformer-circuits.pub/2022/solu
solu(x) = x .* softmax(x)

include("attention.jl")
include("blocks.jl")
include("bigram.jl")
include("circuit.jl")

end
