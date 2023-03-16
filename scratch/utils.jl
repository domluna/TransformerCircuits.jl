using StatsBase

# we can add truncated tokens or we can cutoff tokens that go over the modulo of the block size
function preprocess_data(data, blocksize::Int, tok2idx::Dict{Char,Int64})
    n = length(data)
    n = n - n % blocksize
    return data[1:n]
end

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

function estimate_loss(model, data::Flux.Data.DataLoader; evaliters::Int = 100)
    loss = 0.0
    D = collect(data)
    for _ in 1:evaliters
        x, y = rand(D)
        ŷ = softmax(model(x), dims = 1)
        loss += Flux.Losses.crossentropy(ŷ, y)
    end
    return loss / evaliters
end

function estimate_accuracy(model, x, y; last_token::Bool = false)
    ŷ = softmax(model(x), dims = 1)
    if last_token
        # [:, end, :] is the last token of the sequence
        return mean(argmax(ŷ, dims = 1)[:, end, :] .== argmax(y, dims = 1)[:, end, :])
    end
    return mean(argmax(ŷ, dims = 1) .== argmax(y, dims = 1))
end

function estimate_accuracy(model, data::Flux.Data.DataLoader; kwargs...)
    acc = 0.0
    for (x, y) in data
        acc += estimate_accuracy(model, x, y; kwargs...)
    end
    return acc / length(data)
end

struct Run
    train_losses::Vector{Float64}
    val_losses::Vector{Float64}
    train_accs::Vector{Float64}
    val_accs::Vector{Float64}
end

function Run()
    Run(Float64[], Float64[], Float64[], Float64[])
end

"""
epochs(r::Run)::Int

Returns the number of epochs that have been run.
"""
epochs(r::Run) = length(r.train_losses)

function add_train_eval!(r::Run, train_loss, train_acc)
    push!(r.train_losses, train_loss)
    push!(r.train_accs, train_acc)
end

function add_val_eval!(r::Run, val_loss, val_acc)
    push!(r.val_losses, val_loss)
    push!(r.val_accs, val_acc)
end

function train_model!(
    model,
    optim,
    traindata::Flux.Data.DataLoader;
    valdata::Union{Flux.Data.DataLoader,Nothing} = nothing,
    nepochs::Int = 10,
    evalevery::Int = 1,
    evaliters::Int = 10,
    only_last_token::Bool = false,
    run::Union{Nothing,Run} = nothing,
    early_stop::Union{Nothing,Function} = nothing,
)
    if run === nothing
        run = Run()
    end
    n = epochs(run) + 1
    for epoch in n:n+nepochs
        Flux.train!(
            (m, x, y) -> (loss = Flux.Losses.crossentropy(softmax(m(x), dims = 1), y); loss),
            model,
            traindata,
            optim,
        )
        if epoch % evalevery == 0
            train_loss = estimate_loss(model, traindata; evaliters)
            train_acc = estimate_accuracy(model, traindata; last_token = only_last_token)
            add_train_eval!(run, train_loss, train_acc)
            val_loss::Union{Float64,Nothing} = nothing
            val_acc::Union{Float64,Nothing} = nothing
            if valdata !== nothing
                val_loss = estimate_loss(model, valdata; evaliters)
                val_acc = estimate_accuracy(model, valdata; last_token = only_last_token)
            end
            add_val_eval!(run, val_loss, val_acc)
            @info "Evaluation" epoch train_loss train_acc val_loss val_acc
            if early_stop !== nothing
                if early_stop()
                    break
                end
            end
        end
    end
    return run
end

# given an initial sequence, generate a sequence of length n
# If the sequence is shorter than the blocksize, we pad it with newlines (prepend)
# and then remove the newlines from the output
function generate_text(model, seq::Vector{Int}, blocksize::Int, n::Int, temperature::Float64)
    generated = copy(seq)
    for _ in 1:n
        context_size = min(length(generated), blocksize)
        context = reshape(generated[max(size(generated, 1) - blocksize + 1, 1):end], (context_size, 1))
        y = model(context)
        T = eltype(y)
        output = y[:, end, end] ./ T(temperature)
        idx = StatsBase.sample(1:length(output), ProbabilityWeights(output))
        generated = vcat(generated, idx)
    end
    generated
end
generate_text(model, seq::String, blocksize::Int; n::Int = 1, temperature = 1.0) =
    generate_text(model, encode(seq), blocksize, n, temperature)
generate_text(model, char::Char, blocksize::Int; n::Int = 1, temperature = 1.0) =
    generate_text(model, encode(string(char)), blocksize, n, temperature)

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
