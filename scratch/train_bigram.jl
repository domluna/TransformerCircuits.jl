using Flux
using TransformerCircuits
using StatsBase

# Form the dataset
text = read("../data/input.txt", String)
chars = Set(text)
vocabsize = length(chars)
char2idx = Dict(c => i for (i, c) in enumerate(chars))
idx2char = Dict(i => c for (i, c) in enumerate(chars))

encode(text) = [char2idx[c] for c in text]

decode(encoded) = [idx2char[i] for i in encoded]

encoded_text = encode(text)
split_idx = Int(round(length(encoded_text) * 0.9))
train_data = encoded_text[1:split_idx]
val_data = encoded_text[split_idx+1:end]

blocksize = 8

# we can add truncated tokens or we can cutoff tokens that go over the modulo of the block size

function cutoff_data(data, blocksize::Int)
    n = length(data)
    n = n - n % blocksize
    return data[1:n]
end
train_data = cutoff_data(train_data, blocksize)
val_data = cutoff_data(val_data, blocksize)

function generate_batch(
    encoded_data::Vector{Int64},
    batchsize::Int,
    blocksize::Int,
    vocabsize::Int,
)
    n = length(encoded_data)
    idxs = rand(1:n-blocksize, batchsize)
    x = zeros(Int64, blocksize, batchsize)
    y = zeros(Int64, blocksize, batchsize)
    for i in 1:batchsize
        x[:, i] = encoded_data[idxs[i]:idxs[i]+blocksize-1]
        y[:, i] = encoded_data[idxs[i]+1:idxs[i]+blocksize]
    end
    return x, Flux.onehotbatch(y, 1:vocabsize)
end

X, Y = generate_batch(train_data, length(train_data) ÷ blocksize, blocksize, vocabsize)

batchsize = 128
data = Flux.DataLoader((X, Y); batchsize)

model = Bigram(vocabsize)
optim = Flux.setup(Adam(), model)

nepochs = 10
for epoch in 1:nepochs
    Flux.train!(
        (m, x, y) ->
            (loss = Flux.Losses.crossentropy(m(x), y); println("Loss $loss"); loss),
        model,
        data,
        optim,
    )
end

# given an initial sequence, generate a sequence of length n
function generate_text(model, seq::Vector{Int}, n::Int)
    generated = copy(seq)
    for _ in 1:n
        context = reshape(generated, :, 1)
        y = model(context)
        output = y[:, end, end]
        idx = StatsBase.sample(1:length(output), ProbabilityWeights(output))
        generated = vcat(generated, idx)
    end
    return generated
end

function generate_text(model::Matrix{Float64}, seq::Vector{Int}, n::Int)
    generated = copy(seq)
    for _ in 1:n
        context = generated[end]
        y = model[:, context]
        idx = StatsBase.sample(1:length(y), ProbabilityWeights(y))
        generated = vcat(generated, idx)
    end
    return generated
end

generate_text(model, seq::String; n::Int = 1) = generate_text(model, encode(seq), n)
generate_text(model, char::Char; n::Int = 1) = generate_text(model, encode(string(char)), n)

function standard_bigram_model(text::String)
    chars = Set(text)
    vocabsize = length(chars)
    char2idx = Dict(c => i for (i, c) in enumerate(chars))

    m = zeros(Int, vocabsize, vocabsize)

    for i in 1:length(text)-1
        # access by column
        m[char2idx[text[i+1]], char2idx[text[i]]] += 1
    end

    m = m ./ sum(m, dims = 1)
    return m
end

M = standard_bigram_model(text)

s = join(sort(collect(chars)), "")
e = encode(s)

join(decode(generate_text(M, s, n = 50)), "") |> print
join(decode(generate_text(model, s, n = 50)), "") |> print
