using Flux
using TransformerCircuits

# Form the dataset
text = read("../data/input.txt", String)
chars = Set(text)
vocab_size = length(chars)
char2idx = Dict(c => i for (i, c) in enumerate(chars))
idx2char = Dict(i => c for (i, c) in enumerate(chars))

encode(text) = [char2idx[c] for c in text]

decode(encoded) = [idx2char[i] for i in encoded]

encoded_text = encode(text)
split_idx = Int(round(length(encoded_text) * 0.9))
train_data = encoded_text[1:split_idx]
val_data = encoded_text[split_idx+1:end]

block_size = 8

# we can add truncated tokens or we can cutoff tokens that go over the modulo of the block size

function cutoff_data(data, block_size::Int)
    n = length(data)
    n = n - n % block_size
    return data[1:n]
end
train_data = cutoff_data(train_data, block_size)
val_data = cutoff_data(val_data, block_size)

function generate_batch(encoded_data::Vector{Int64}, batch_size::Int, block_size::Int)
    n = length(encoded_data)
    idxs = rand(1:n-block_size, batch_size)
    x = zeros(Int64, block_size, batch_size)
    y = zeros(Int64, block_size, batch_size)
    for i in 1:batch_size
        x[:, i] = encoded_data[idxs[i]:idxs[i]+block_size-1]
        y[:, i] = encoded_data[idxs[i]+1:idxs[i]+block_size]
    end
    return x, y
end

X, Y = generate_batch(train_data, length(train_data)  ÷ block_size, block_size)

batch_size = 4

data = Flux.DataLoader((X, Y), batchsize = batch_size)

model = Bigram(vocab_size)
o = model(x)

optim = Flux.setup(Adam(), model)
for epoch in 1:1000
    Flux.train!((m, x, y) -> Flux.Losses.cross_entropy(m(x), y), model, data, optim)
end
