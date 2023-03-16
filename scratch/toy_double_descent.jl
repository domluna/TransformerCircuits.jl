using TransformerCircuits
using Flux

include("utils.jl")

vocabsize = size(trainY, 1)
blocksize = size(trainX, 1)
dembed = 128
circ = Circuit(vocabsize, blocksize, dembed; nheads = 4);
opt = Flux.setup(AdamW(1e-3), circ);

# train_model!(circ, opt, traindata; nepochs = 10, evaliters = 1)
# # it's a single batch size so 1 evaliter is the entire dataset
# train_loss = estimate_loss(circ, traindata, evaliters = 1)
# @info "Training loss" train_loss
# val_loss = estimate_loss(circ, valdata, evaliters = 1)
# @info "Validation loss" val_loss
