module TransformerCircuits

export Circuit, BiGram

using Flux

const A3{T} = AbstractArray{T,3} where {T<:Union{Float64,Float32,Float16}}

include("attention.jl")
include("blocks.jl")
include("bigram.jl")
include("circuit.jl")

end
