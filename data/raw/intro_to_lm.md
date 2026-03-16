# Introduction to Language Models

A language model is a probability distribution over sequences of words or tokens.
Given a sequence of tokens, a language model assigns a probability to the next token.
This simple formulation powers everything from autocomplete to large-scale AI assistants.

## How Language Models Work

Language models operate on text that has been converted to tokens.
Tokenisation is the process of splitting text into subword units.
Each subword is mapped to an integer called a token ID.
The model processes these integer sequences, not raw characters.

### Tokenisation

Byte-Pair Encoding, or BPE, is a popular tokenisation algorithm.
It starts with individual characters as the vocabulary.
Then it iteratively merges the most frequent adjacent pair of tokens.
This continues until the vocabulary reaches a desired size.
Common words become single tokens while rare words are split into pieces.

## The Transformer Architecture

The transformer was introduced in the 2017 paper "Attention Is All You Need."
It replaced recurrent networks for sequence modelling tasks.
The key innovation was the self-attention mechanism.

### Self-Attention

Self-attention allows each position in a sequence to attend to all other positions.
For each token, the model computes three vectors: Query, Key, and Value.
The attention score between two positions is the dot product of their Query and Key vectors, scaled by the square root of the dimension.
Scores are passed through a softmax to produce attention weights.
Each position's output is a weighted sum of Value vectors.

### Multi-Head Attention

Instead of one set of Q, K, V projections, transformers use multiple attention heads.
Each head learns to attend to different aspects of the input.
The outputs of all heads are concatenated and projected back to the model dimension.

### Feed-Forward Networks

After the attention layer, each position is processed independently by a feed-forward network.
This consists of two linear layers with a non-linear activation function between them.
The hidden dimension is usually four times the model's embedding dimension.
GELU activation is commonly used in modern transformers.

### Layer Normalisation

Layer normalisation is applied before each sub-layer in the transformer block.
This is called pre-norm and leads to more stable training than post-norm.
It normalises the mean and variance of activations across the feature dimension.

## Training Language Models

Language models are trained with next-token prediction.
Given a sequence of tokens, the model predicts the next token at every position.
The loss is the average cross-entropy across all positions.

### Optimisation

AdamW is the standard optimiser for transformer training.
It decouples weight decay from the gradient update.
Weight decay is applied only to weight matrices, not to biases or normalisation parameters.
A cosine learning rate schedule with a short warm-up period is widely used.

### Gradient Clipping

Gradients can occasionally become very large, causing unstable parameter updates.
Gradient clipping scales down gradients whose norm exceeds a threshold.
A typical threshold is 1.0.

### Gradient Accumulation

When memory is limited, gradient accumulation simulates a larger batch.
Instead of updating after every batch, gradients are summed over several steps.
The optimiser step is taken after the desired number of accumulation steps.

## Overfitting and Regularisation

Overfitting occurs when a model learns the training data too well.
It memorises specific examples instead of learning general patterns.
The result is poor performance on unseen data.

### Dropout

Dropout randomly sets a fraction of activations to zero during training.
This prevents neurons from co-adapting and forces the network to learn robust features.
A dropout rate of 0.1 to 0.3 is typical for transformer models.

### Weight Tying

In language models, the input embedding matrix and the output projection matrix operate in the same space.
Tying their weights (sharing parameters) reduces the model size.
It also acts as a regulariser and often improves performance.

### Early Stopping

Monitoring validation loss during training helps detect overfitting.
When validation loss stops decreasing, training can be stopped.
The checkpoint with the lowest validation loss is the best model.

## Inference and Generation

At inference time, the model generates text autoregressively.
Starting from a prompt, it predicts the next token, appends it, and repeats.
This continues until a stop token is generated or a maximum length is reached.

### Sampling Strategies

Greedy decoding always picks the highest probability next token.
This produces repetitive, dull text.

Temperature scaling adjusts the sharpness of the probability distribution.
Lower temperature produces more focused outputs.
Higher temperature produces more varied, creative outputs.

Top-k sampling restricts the choice to the k highest probability tokens.
Nucleus sampling, or top-p, restricts to the smallest set of tokens whose cumulative probability exceeds p.
Both methods improve diversity while maintaining coherence.

## Small Language Models

Large language models require significant compute for training and inference.
Small language models with 10 to 100 million parameters can be trained on a single consumer machine.
They are useful for learning, experimentation, and domain-specific applications with limited data.

Training a small model from scratch teaches the fundamental concepts that scale to larger systems.
The same architecture, training loop, and evaluation strategy applies regardless of model size.

## Summary

Language models assign probabilities to token sequences.
Transformers use self-attention to model relationships across the entire context window.
Training uses next-token prediction with cross-entropy loss.
Regularisation through dropout, weight tying, and gradient clipping improves generalisation.
Small models can be trained locally without specialised hardware.
