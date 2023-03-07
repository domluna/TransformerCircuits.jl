using TransformerCircuits
using Flux

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
num2tok = Dict(i - 1 => c for (i, c) in enumerate(alphabet))
tok2num = Dict(values(num2tok) .=> keys(num2tok))

function create_dataset_binop_with_mod(
    op::Function,
    modn::Int,
    datasetsize::Int = 1024,
)
    toks = alphabet[1:modn+1]
    push!(toks, string(Symbol(op)))
    push!(toks, "=")
    push!(toks, " ")
    tok2idx = Dict(c => i for (i, c) in enumerate(toks))
    idx2tok = Dict(i => c for (i, c) in enumerate(toks))

    xs = Vector{Int}[]
    ys = Vector{Int}[]
    for _ in 1:datasetsize
        a = rand(0:modn)
        b = rand(0:modn)
        c = (op(a, b)) % modn
        s = "$(num2tok[a]) $op $(num2tok[b]) = $(num2tok[c])"
        # encode
        enc = [tok2idx[string(c)] for c in s]
        push!(xs, enc[1:end-1])
        push!(ys, enc[2:end])
    end

    # TODO: matrix
    X = zeros(Int, (length(xs[1]), datasetsize))
    Y = zeros(Int, (length(ys[1]), datasetsize))
    for i in 1:datasetsize
        X[:, i] = xs[i]
        Y[:, i] = ys[i]
    end
    return (X, Flux.onehotbatch(Y, 1:length(tok2idx))), tok2idx, idx2tok
end

modn = 33
data, tok2idx, idx2tok = create_dataset_binop_with_mod(+, modn)

# split the dataset and shuffle it
X, Y = data
n = Int(round(size(X, 2) * 0.5))
trainX, trainY = X[:, 1:n], Y[:, :, 1:n]
valX, valY = X[:, n+1:end], Y[:, :, n+1:end]

traindata = Flux.DataLoader((trainX, trainY), batchsize = size(trainX, 2))
valdata = Flux.DataLoader((valX, valY), batchsize = size(valX, 2))

vocabsize = size(trainY, 1)
blocksize = size(trainX, 1)
dembed = 128
circ = Circuit(vocabsize, blocksize, dembed; nheads = 4);
opt = Flux.setup(AdamW(1e-3), circ);

train_model!(circ, traindata, opt; nepochs = 10, evaliters = 1)
# it's a single batch size so 1 evaliter is the entire dataset
train_loss = estimate_loss(circ, traindata, evaliters = 1)
@info "Training loss" train_loss
val_loss = estimate_loss(circ, valdata, evaliters = 1)
@info "Validation loss" val_loss
