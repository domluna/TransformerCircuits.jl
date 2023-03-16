# Experimenting with n-gram models
# The bigram model learns the probability of a character given the previous character
# and approximates the ideal model which can be calculated by counting the number of times
# a character transition to another character and dividing by the total number of times it transitions.
#
# Once you add more layers the model begins to use more context.
using Flux
using TransformerCircuits
using Plots

include("utils.jl")

function standard_bigram_model(text::String, tok2idx::Dict{Char,Int})
    vocabsize = length(tok2idx)
    m = zeros(Int, vocabsize, vocabsize)

    for i in 1:length(text)-1
        # access by column
        m[tok2idx[text[i+1]], tok2idx[text[i]]] += 1
    end

    m = m ./ sum(m, dims = 1)
    return m
end

# Form the dataset
text = read("data/input.txt", String)
chars = Set(text)

# TODO: add special tokens ???
# available - @, #
# start_token = '@'
# stop_token = '#'

vocabsize = length(chars)
tok2idx = Dict(c => i for (i, c) in enumerate(chars))
idx2tok = Dict(i => c for (i, c) in enumerate(chars))

encode(text) = [tok2idx[c] for c in text]
decode(encoded_text) = [idx2tok[i] for i in encoded_text]

# constants
batchsize = 128
blocksize = 512
vocabsize = length(tok2idx)

encoded_text = encode(text)
split_idx = Int(round(length(encoded_text) * 0.9))
train_data = encoded_text[1:split_idx]
val_data = encoded_text[split_idx+1:end]

# Might not need this if we use padding
train_data = cutoff_data(train_data, blocksize)
val_data = cutoff_data(val_data, blocksize)

X, Y = generate_batch(train_data, length(train_data) ÷ blocksize, blocksize, vocabsize)
train_data = Flux.DataLoader((X, Y); batchsize, shuffle = true)
X, Y = generate_batch(val_data, length(val_data) ÷ blocksize, blocksize, vocabsize)
val_data = Flux.DataLoader((X, Y); batchsize)

function standard_bigram_model(text::String, tok2idx::Dict{Char,Int})
    vocabsize = length(tok2idx)
    m = zeros(Int, vocabsize, vocabsize)

    for i in 1:length(text)-1
        # access by column
        m[tok2idx[text[i+1]], tok2idx[text[i]]] += 1
    end

    m = m ./ sum(m, dims = 1)
    return m
end

M = standard_bigram_model(text, tok2idx)

s = join(sort(collect(chars)), "")
e = encode(s)

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

bigram_model = BiGram(vocabsize);
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

ngram_model = Circuit(vocabsize, blocksize, 32; nheads = 4);
ngram_optim = Flux.setup(AdamW(), ngram_model);
train_model!(ngram_model, ngram_optim, train_data; nepochs = 10)
train_loss = estimate_loss(ngram_model, train_data, evaliters = 100)
@info "" train_loss
val_loss = estimate_loss(ngram_model, val_data, evaliters = 100)
@info "" val_loss

circ = Circuit(vocabsize, blocksize, 128; nheads = 8);
opt = Flux.setup(AdamW(5e-4), circ);
train_model!(circ, opt, train_data; nepochs = 10)
train_loss = estimate_loss(circ, train_data, evaliters = 50)
@info "Training loss" train_loss
val_loss = estimate_loss(circ, val_data, evaliters = 50)
@info "Validation loss" val_loss
# more than 1 head leads to a better loss
#
# 2 layers decreases the loss more.
# julia> train_model!(circ, opt, train_data; nepochs = 10)
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

circ2 = Circuit(vocabsize, blocksize, 128; nheads = 16);

# TODOS:
# - implement larger context
# - add <START> <END> tokens (required?)
# measure 500th - 50th token loss
# [a][b][a][b]....[a] -> [b]
# [a*][b*][a*][b*]....[a] -> [b] ... what would b be?
#
# The core idea of in-context learning is the generation changes dependent on the input.
#
# as an example we can construct bigram plots for each character and then create a context in which
# the most probable character in the bigram sense is clearly not the most probable character in the
# context sense.

join(decode(generate_text(circ2, s, 50)), "") |> print
