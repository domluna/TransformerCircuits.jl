# Note 1:
#
# This gets to 100% on the train set quickly but then on the validation set highest I've got is 65% or so
# after 150k epochs, which is quite a bit less than the 10^6 in some of the experiments. I'm not on a GPU
# right now so the iterations take much longer.
# Revisit this when I have access to my desktop again.
#
# Note 2:
#
# After adding "% modn" to the string the model gets to 75% accuracy on the last token. It gets to 100% on the
# train set super quick.
#
# Interestingly enough the validation curve is pretty different from what's reported in the paper. It consistently
# goes up but the climb just becomes slower and slower rather than it being flat for a long period and then shooting up
# like a rocket.
#
# Note 3:
#
# I was using 512 as the maximum training set size but in the paper they use the entire dataset so for mod 97 that would
# be 9604 exammples (98 * 98). So it's 50% train/val split. 512 is the minibatch size (I misread that :( ).
#
# When I did this and had the operator and "% modn" as part of the encoding the validation accuracy hit 100% after 1000 epochs
# which is way sooner than in the paper. After reading more carefully "∘" is used to note the operator, meaning the actual operation
# is never shown to the model.
#
# Note 4:
#
# After making this change on the +, mod 97 dataset the validation accuracy is still 100% after 1000 epochs. With a 50/50 train/val split
# With a 30/70 train/val split the validation accuracy is >= 99% after 39900 epochs. So 40x longer to hit 99%.
#
# Note 5:
#
# a ÷ b mod 97
# 50/50 train/val split >= 99% validation accuracy after 1100 epochs
# In the paper they use optimizations steps, which I think would be each minibatch evaluation. So 1100 * 10 (train dataset batches) = 11000 steps.
# That's 10^4 which is an order of magnitude less than the 10^5 steps in the paper before they see any movement in validation accuracy going up.
# And there's movement before 100 epochs are done.
#
# UPDATE: It turns out there was an error dividing by 0 when the dataset was created so the experiments were on the plus dataset not
# the division dataset. It seems the division dataset hits 90% validation accuracy very quickly and then stays there for a long time.
# I wonder if it's because there are symbol(s) that are never used in the validation set. -- No, that's not the case. We generate data a<-1..modn-1, b<-1...mod-n1. If this is not shuffled, i.e. we split in order, validation accuracy is significantly worse. 30-40% vs 90-95%. Interestingly the validation loss curve goes up.
#
# We're still doing seq2seq loss right now but this problem is more akin to predicting the final output given the prior sequence. The entire sequence loss doesn't make sense frankly since for a<op>b a<op> gives no insight into what b will be.
#
#
# Note 6:
# x^3 + xy^2 + y mod 97
# 50/50 split after 100k epochs gives 2% validation accuracy
# 95/5 split after 100k epochs gives 5% validation accuracy
# So this is the same as the paper. Note sure if they mention the accuracy but they mention it doesn't generalize to this problem.
#
# Note 7:
# seq2val is working super well. We're now getting the results we expect. The validation accuracy increases after a certain point, where the validation starts to decrease. Prior to this point the validation loss would steadily increase.
#
# Note 8:
#
# (x, y) -> isodd(y) ? (x ÷ y) % modn : (x - y) % modn
#
# This is the same as the paper. The model can generalize to this problem as well even though it's asymmetrical, however, it does take significantly longer.
#
# Note 9:
#
# It could be due to a particular seed but the validation accuracy does not necessarily stay at 0% and then start going up, it might start anywhere from 0-40% and then plateau for a long while before going up.
using TransformerCircuits
using Flux
using Random
using Serialization
using Plots
using BSON

include("utils.jl")

# Form the dataset
# functions of the form
# a <binop> b = c

# Assuming each example is of the form "a <binop> b = c" (9 tokens)
# and each batch is 512 examples, and this is repeated for 10^5 steps
# that's 460_800_000 tokens. Which is not the 2.5B-5B token range
# required for the phase change.
#
# The mentioned phase change refers to in-context learning but for
# the grokking dataset this ability would not matter since the context length
# is so short.
#
# However 10^5 is the lower end and for most experiments the total training time is % * 10^5
# or 10^6 steps which would fall in the 2.5-5B token range.
#
alphabet = string.(vcat('a':'z', 'A':'Z', Char.(1024:2048)))

# number ranges are 1:modn-1 and -1:-modn-1
function create_dataset_binop_with_mod(f::Function, modn::Int)
    nums = collect(-modn:modn)
    num2tok = Dict{Int,String}()
    for (i, n) in enumerate(nums)
        num2tok[n] = alphabet[i]
    end

    tok2num = Dict(values(num2tok) .=> keys(num2tok))
    toks = collect(values(num2tok))
    push!(toks, "=")
    push!(toks, "∘")
    tok2idx = Dict(c => i for (i, c) in enumerate(toks))
    idx2tok = Dict(i => c for (i, c) in enumerate(toks))

    data = Vector{Int}[]
    for a in 1:modn
        for b in 1:modn
            c = f(a, b)
            # the operation is hidden from the model
            # all that's is the inputs and output
            s = "$(num2tok[a])∘$(num2tok[b])=$(num2tok[c])"
            # encode
            enc = [tok2idx[string(c)] for c in s]
            push!(data, enc)
        end
    end
    Random.shuffle!(data)

    X = zeros(Int, (length(data[1]) - 1, length(data)))
    y = zeros(Int, length(data))
    for (i, enc) in enumerate(data)
        X[:, i] = enc[1:end-1]
        y[i] = enc[end]
    end
    return (X, Flux.onehotbatch(y, 1:length(tok2idx))), tok2idx, idx2tok
end

function decode(x::Vector{Int})
    s = ""
    for i in 1:size(x, 1)
        tok = idx2tok[x[i]]
        if get(tok2num, tok, nothing) !== nothing
            s *= "$(tok2num[tok])"
        else
            s *= tok
        end
    end
    return s
end

function decode(model, x::AbstractMatrix{Int})
    o = Flux.onecold(model(x), 1:vocabsize)
    outputs = String[]
    for i in 1:size(o, 2)
        push!(outputs, decode(vcat(x[:, i], o[end, i])))
    end
    return outputs
end

modn = 97
# +
# data, tok2idx, idx2tok = create_dataset_binop_with_mod((a, b) -> (a + b) % modn, modn)
# -
# data, tok2idx, idx2tok = create_dataset_binop_with_mod((a, b) -> (a - b) % modn, modn)

# division
data, tok2idx, idx2tok =
    create_dataset_binop_with_mod((a, b) -> isodd(b) ? (a ÷ b) % modn : (a - b) % modn, modn)

# the paper says x^3 + xy^2 + y mod 97 did not lead to generalization even with a 95/5 split
# data, tok2idx, idx2tok = create_dataset_binop_with_mod((a, b) -> (a^3 + a * b^2 + b) % modn, modn)

X, Y = data;
trainfrac = 0.5;
N = size(X, 2);
n = Int(round(N * trainfrac));
trainX, trainY = X[:, 1:n], Y[:, 1:n];
valX, valY = X[:, n+1:N], Y[:, n+1:N];

trainX = trainX |> gpu;
trainY = trainY |> gpu;
valX = valX |> gpu;
valY = valY |> gpu;

train_batchsize = min(512, size(trainX, 2))
val_batchsize = min(512, size(valX, 2))
traindata = Flux.DataLoader((trainX, trainY), batchsize = train_batchsize, shuffle = true);
valdata = Flux.DataLoader((valX, valY), batchsize = val_batchsize);

vocabsize = size(trainY, 1)
blocksize = size(trainX, 1)
# paper used 128 for embedding size and 4 heads
dembed = 128
nheads = 4
nlayers = 2
circ = Circuit(vocabsize, blocksize, dembed; nheads, nlayers) |> gpu;
opt = Flux.setup(AdamW(3e-4), circ);

@info """
$(N) total examples for mod $modn
$(size(trainX, 2)) training examples
$(size(valX, 2)) validation examples
$(size(trainX, 1)) tokens per example
$(length(tok2idx)) tokens in the vocabulary
"""

run = Run()
evalevery = 10
train_model!(
    circ,
    opt,
    traindata;
    nepochs = 10_000,
    evaliters = 10,
    evalevery = evalevery,
    valdata = valdata,
    seq2val = true,
    early_stop = () -> begin
        # stop if the validation accuracy is >= 0.99
        accuracy_metric(circ, valdata; seq2val = true) >= 0.99
    end,
    run = run,
)

name = "hardfunction_97"
serialize("runs/$(name)_run.jls", run)

nsteps = length(run.train_losses) * evalevery * length(traindata)
xsteps = 1:(nsteps÷length(run.train_losses)):nsteps
# set the ticks for every 100 if there are 1000 or less steps, otherwise every 1000

plot(xsteps, run.train_accs, label = "Training accuracy", linewidth = 2)
plot!(xsteps, run.val_accs, label = "Validation accuracy", linewidth = 2)
ylims!((0.0, 1.1))
# xlims!((0, xsteps[end]))

# add titles and labels
title!("Training and validation accuracy")
xlabel!("Optimization Steps")
ylabel!("Accuracy")
# show the plot
display(plot!)

savefig("images/$(name)_accuracy.png")

# create a new plot for the losses
plot(xsteps, run.train_losses, label = "Train loss")
plot!(xsteps, run.val_losses, label = "Val loss")

# add title and axis labels for the loss plot
title!("Training and validation loss")
xlabel!("Optimization Steps")
ylabel!("Loss")

# save the loss plot
savefig("images/$(name)_loss.png")

BSON.@save "models/circ$(nlayers)-$(nheads)-$(dembed)_$(name)_model.bson" circ
