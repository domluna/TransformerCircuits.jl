# size of (sequence_length, batch_size) or (1, sequence_length, batch_size)
# embeds to (dembed, sequence_length, batch_size) or (dembed, sequence_length * batch_size)

struct FFN
    d1::Dense
    d2::Dense
    drop1::Dropout
    drop2::Dropout
end
Flux.@functor FFN

FFN(dembed::Int, dff::Int, dropout_prob::Float32, dff_activation) = FFN(
    Dense(dembed, dff, dff_activation),
    Dropout(dropout_prob),
    Dense(dff, dembed),
    Dropout(dropout_prob),
)

function (ffn::FFN)(inp)
    sz = size(inp)
    x = ffn.d1(reshape(inp, sz[1], :))
    x = fnn.drop1(x)
    x = ffn.d2(x)
    x = fnn.drop2(x)
    reshape(x, sz)
end

struct EncoderBlock{A,F}
    attn::A
    ffn::F
    drop::Dropout
    ln1::LayerNorm
    ln2::LayerNorm
end
Flux.@functor EncoderBlock

function EncoderBlock(nheads::Int, dembed::Int, dff::Int, dff_activation::Function, dropout_prob::Float32)
    EncoderBlock(
        SelfAttention(dembed, nheads),
        FFN(dembed, dff, dropout_prob, dff_activation),
        Dropout(dropout_prob),
    )
end

EncoderBlock(dembed::Int, nheads::Int) = EncoderBlock(dembed, dembed, nheads)

function EncoderBlock(attn::MultiHeadAttention, ffn::FFN, drop::Dropout)
    ln1 = LayerNorm(size(attn.O, 1))
    ln2 = LayerNorm(size(ffn.d2.W, 1))
    EncoderBlock(attn, ffn, drop, ln1, ln2)
end

function (b::EncoderBlock)(inp)
    x = b.attn(b.ln1(inp))
    x = b.drop(x) + inp
    b.ffn(b.ln2(x)) + x
end

struct Encoder
    blocks::Chain
    ln::LayerNorm
end
Flux.@functor Encoder

function Encoder(;
    nblocks = 1,
    nheads = 12,
    dembed = 784,
    dff = dembed * 4,
    dff_activation = gelu,
    dropout_prob::Float32 = 0.1
)
    blocks = Chain([
        EncoderBlock(nheads, dembed, dff, dropout_prob, dff_activation)
        for _ = 1:nblocks
    ]...)
    ln = LayerNorm(dembed)
    Encoder(blocks, ln)
end


struct GPT
    position_embedding::Any
    token_embedding::Any
    encoder::Any

    drop::Dropout
end
Flux.@functor GPT

function GPT(;
    nblocks = 1,
    nheads = 12,
    dembed = 784,
    dff = dembed * 4,
    dff_activation = gelu,
    dropout_prob::Float32 = 0.1
)
    GPT(
        Embedding(dembed, dembed),
        Embedding(dembed, dembed),
        Encoder(; nblocks, nheads, dembed, dff, dff_activation, dropout_prob),
        Dropout(dropout_prob),
    )
end

function (gpt::GPT)(inp)
    x = gpt.embedding(inp)
    x = gpt.drop(x)
    x = gpt.encoder(x)
    x
end
