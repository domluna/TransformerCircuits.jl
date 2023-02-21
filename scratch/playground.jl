using Flux
using TransformerCircuits
using StatsBase
using Plots

# Form the dataset
text = read("data/input.txt", String)
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

blocksize = 128
vocabsize = length(char2idx)

# we can add truncated tokens or we can cutoff tokens that go over the modulo of the block size
function cutoff_data(data, blocksize::Int)
    n = length(data)
    n = n - n % blocksize
    return data[1:n]
end
train_data = cutoff_data(train_data, blocksize)
val_data = cutoff_data(val_data, blocksize)

function generate_batch(encoded_data::Vector{Int64}, batchsize::Int, blocksize::Int, vocabsize::Int)
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

batchsize = 128
X, Y = generate_batch(train_data, length(train_data) ÷ blocksize, blocksize, vocabsize)
train_data = Flux.DataLoader((X, Y); batchsize)
X, Y = generate_batch(val_data, length(val_data) ÷ blocksize, blocksize, vocabsize)
val_data = Flux.DataLoader((X, Y); batchsize)

function estimate_loss(model, data::Flux.Data.DataLoader; evaliters::Int = 100)
    loss = 0.0
    D = collect(data)
    for _ in 1:evaliters
        x, y = rand(D)
        ŷ = model(x)
        loss += Flux.Losses.crossentropy(ŷ, y)
    end
    return loss / evaliters
end

function train_model!(model, data::Flux.Data.DataLoader, optim; nepochs::Int = 10)
    for _ in 1:nepochs
        Flux.train!(
            (m, x, y) -> (loss = Flux.Losses.crossentropy(m(x), y); loss),
            model,
            train_data,
            optim,
        )
        @info "Training loss" estimate_loss(model, train_data, evaliters = 30)
    end
end

# End of data setup
# Train the model

bigram_model = Bigram(vocabsize)
bigram_optim = Flux.setup(AdamW(), model)
train_model!(bigram_model, train_data, bigram_optim, nepochs = 10)
# julia> estimate_loss(model, train_data, evaliters=100)
# 2.445962369441986
#
# julia> estimate_loss(model, val_data, evaliters=100)
# 2.4857913732528685

# It's hard to do any better than this since the prediction is only based on the most recent
# previous character. Vowels or whitespace will be a probable next output for many characters.

# given an initial sequence, generate a sequence of length n
# If the sequence is shorter than the blocksize, we pad it with newlines (prepend)
# and then remove the newlines from the output
function generate_text(model, seq::Vector{Int}, n::Int)
    generated = copy(seq)
    m = blocksize - length(generated)
    if length(generated) < blocksize
        generated = vcat(encode(repeat('\n', m)), generated)
    end
    for _ in 1:n
        context_size = min(length(generated), blocksize)
        context = reshape(generated[max(size(generated, 1) - blocksize + 1, 1):end], (context_size, 1))
        y = model(context)
        output = y[:, end, end]
        idx = StatsBase.sample(1:length(output), ProbabilityWeights(output))
        generated = vcat(generated, idx)
    end
    length(generated) < blocksize ? generated[end-blocksize+1:end] : generated
end
generate_text(model, seq::String; n::Int = 1) = generate_text(model, encode(seq), n)
generate_text(model, char::Char; n::Int = 1) = generate_text(model, encode(string(char)), n)

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
join(decode(generate_text(bigram_model, s, n = 50)), "") |> print

sorted_chars = join(decode(collect(1:vocabsize)), "")
char_list = split(sorted_chars, "")

# plotlyjs()
# Plots.heatmap(M, xticks=(1:vocabsize, char_list), yticks=(1:vocabsize, char_list), aspect_ratio=1, size=(1200, 800), xtickfont= font(10), ytickfont=font(10), hover=char_list)
Plots.heatmap(
    M,
    xticks = (1:vocabsize, char_list),
    yticks = (1:vocabsize, char_list),
    aspect_ratio = 1,
    size = (1200, 800),
    xtickfont = font(10),
    ytickfont = font(10),
)

# same because context size doesn't matter
o1 = bigram_model(reshape(encode(sorted_chars), (1, :)))
o2 = bigram_model(reshape(encode(sorted_chars), (:, 1)))
e1 = o1[:, 1, 1]
e2 = reshape(o2[:, 1, :], :)
e1 ≈ e2 # true

# reshape so it's the same as the bigram model
o2 = reshape(o2, (vocabsize, vocabsize))
Plots.heatmap(
    o2,
    xticks = (1:vocabsize, char_list),
    yticks = (1:vocabsize, char_list),
    aspect_ratio = 1,
    size = (1200, 800),
    xtickfont = font(10),
    ytickfont = font(10),
)

# n-gram model
# all previous context is used
# self attention seems to allow the context to communicate

ngram_model = NGram(vocabsize, blocksize, 32; nheads = 4);
ngram_optim = Flux.setup(AdamW(), ngram_model);
train_model!(ngram_model, train_data, ngram_optim; nepochs = 10)
train_loss = estimate_loss(ngram_model, train_data, evaliters = 100)
@info "" train_loss
val_loss = estimate_loss(ngram_model, val_data, evaliters = 100)
@info "" val_loss

circ = TransformerCircuits.Circuit(vocabsize, blocksize, 128; nheads = 8);
opt = Flux.setup(AdamW(5e-4), circ);
train_model!(circ, train_data, opt; nepochs = 10)
train_loss = estimate_loss(circ, train_data, evaliters = 50)
@info "Training loss" train_loss
val_loss = estimate_loss(circ, val_data, evaliters = 50)
@info "Validation loss" val_loss
# more than 1 head leads to a better loss
#
# 2 layers decreases the loss more.
# julia> train_model!(circ, train_data, opt; nepochs = 10)
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.994046966234843
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.8851592381795248
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.796162517865499
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.7615622440973917
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.7299057960510253
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.7040851791699727
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.7044827143351238
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.686379869778951
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.6706287344296773
# ┌ Info: Training loss
# └   estimate_loss(model, train_data, evaliters = 30) = 1.6709984064102172
# 
# julia>
# 
# julia> val_loss = estimate_loss(circ, val_data, evaliters = 50)
# 1.876984875202179
