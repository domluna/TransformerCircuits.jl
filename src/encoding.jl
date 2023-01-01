function sinusoidal_encoding(seqlen::Int, dembed::Int)
    pe = zeros(Float32, dembed, seqlen)
    r = LinRange{Float32}(0, seqlen, seqlen)

    @inbounds for i in 1:2:dembed
        v = r ./ (10000 .^ ((i - 1) / dembed))
        pe[i, :] = cos.(v)
        pe[i+1, :] = sin.(v)
    end

    pe
end
