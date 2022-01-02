function sinusoidal_encoding(sequence_len::Int, dembed::Int)
    pe = zeros(Float32, dembed, sequence_len)
    r = LinRange{Float32}(0, sequence_len, sequence_len)

    @inbounds for i = 1:2:dembed
        v = r ./ (10000 .^ ((i - 1) / dembed))
        pe[i, :] = cos.(v)
        pe[i+1, :] = sin.(v)
    end

    pe
end
