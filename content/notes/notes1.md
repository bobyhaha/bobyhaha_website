---
title: "The Residual Stream: A Unified View of How Transformers Think"
date: 2026-04-12
---

Modern transformer models look complicated — stacked attention heads, feed-forward networks, layer norms, skip connections. But underneath the engineering, there is a surprisingly elegant story about what all of these pieces are actually doing: **gradually refining a shared, evolving representation of meaning**, one additive update at a time.

## The Residual Stream as a Shared Scratchpad

At the heart of every transformer is the **residual stream** — a vector for each token position that persists across every layer of the network. You can think of it as a shared whiteboard that every layer reads from and writes to.

The key insight, formalized by Elhage et al. in Anthropic's 2021 paper *"A Mathematical Framework for Transformer Circuits"*, is that neither attention nor MLP layers replace the residual stream — they only *add* to it. Formally, each layer's contribution is:

$$x_{\text{new}} = x_{\text{old}} + \text{Layer}(x_{\text{old}})$$

This is the skip connection. It means that if a layer has nothing useful to contribute, it can simply output a near-zero vector and leave the stream untouched. Every piece of information written to the stream earlier remains available downstream. The entire forward pass of a transformer is, in this view, a sequence of small, targeted edits to a shared memory.

A useful metaphor: think of the residual stream as a **tree with branches**, where each token is a branch. The gardeners — attention and MLP layers — progressively add and prune information from each branch as the tree grows taller through the layers. Growth is accumulation, not replacement.

## Attention: Exchanging Information Globally, in a Linear Way

Attention is the mechanism by which tokens talk to each other. For each token, it does the following:

1. **Compute a query vector** $q = W_Q \cdot x$ — encoding "what am I looking for?"
2. **Compute key vectors** $k_j = W_K \cdot x_j$ for every other token $j$ — encoding "what do I offer?"
3. **Score all keys against the query**: $\text{score}_j = q \cdot k_j$, then softmax to get attention weights $\alpha_j$
4. **Read out a value**: $v_j = W_V \cdot x_j$ for each token, then form the weighted sum $\sum_j \alpha_j v_j$
5. **Project and add**: pass through $W_O$ and add the result back to the residual stream

Every step here is linear (aside from the softmax). The attention head asks: *given the current context, which directions in representation space should I amplify?* The answer is a weighted linear combination of value vectors — each a linear projection of some other token's state. This is how attention achieves **global information exchange**: a token at position 500 can directly read from position 1, with the weight determined entirely by content. The modification written back is always a linear function of the input.

## MLP Layers: Memory Without Communication

If attention is the social layer — gathering information across the sequence — the MLP is the introspective layer: it works on each token's representation *in isolation*, never looking at neighboring tokens.

The MLP applies two linear transformations with a non-linearity between them:

$$\text{MLP}(x) = W_{\text{out}} \cdot f(W_{\text{in}} \cdot x)$$

where $f$ is ReLU or GELU. For a single token:

1. $W_{\text{in}} \cdot x$ computes dot products between the token's representation and the rows of $W_{\text{in}}$ — each is a **score** measuring how much the current state matches a stored pattern.
2. $f(\cdot)$ applies a non-linearity. With ReLU, only scores above zero survive — the neuron either fires or doesn't.
3. $W_{\text{out}}$ reads off a linear combination of its **columns**, weighted by those surviving scores. Each column is a **value vector** — a direction in representation space to be added.

This is precisely the structure of a **key-value memory lookup**, as identified by Geva et al. in *"Transformer Feed-Forward Layers Are Key-Value Memories"* (2020). The rows of $W_{\text{in}}$ are keys; the columns of $W_{\text{out}}$ are values. The non-linearity is a soft threshold determining which stored facts are relevant.

No information is exchanged between tokens here. The MLP processes one token at a time, looks up its internal memory, and adds the appropriate knowledge vector back. This suggests a natural functional division: attention assembles context-dependent information from other tokens, and the MLP then processes the enriched representation against its store of world knowledge — adding relevant facts and linguistic regularities compressed into its weights during training.

## MLP as Knowledge Storage — and as a Refinement Filter

The key-value memory interpretation has a striking implication: MLP weights are not just computational machinery — they are **knowledge storage**. When a model learns that "Paris is the capital of France," this fact is encoded in the rows and columns of the MLP weight matrices. When "Paris" appears in context, the appropriate keys fire, and value vectors corresponding to "France," "capital," and related concepts are added to the residual stream.

Because the MLP operates *after* attention in each block, it sees a representation already enriched with cross-token context. The MLP is therefore not just retrieving raw stored facts — it is processing the *attention-modified* representation. This gives it the structure of a **refinement filter**: it can amplify what attention found useful, suppress what is irrelevant, and add deeper implications that emerge from the combination of stored knowledge and in-context information. MLP layers may also perform **compression**, projecting large-dimensional context-dependent representations toward more compact, task-relevant subspaces.

## Layer Norm and the Loss Landscape

Without normalization, a transformer faces a difficult optimization problem. Because every layer writes into the same residual stream with no natural scale constraint, gradients can compound across layers — some weight directions grow enormous, others shrink toward zero. The model has access to an unbounded search space.

**Layer normalization** imposes a geometric constraint: it projects the residual stream onto a manifold of fixed norm, restricting the reachable representations at each layer. This is analogous to ideas in optimization on curved manifolds — by restricting the search space to a manifold, gradient descent becomes more stable and the loss landscape smoother.

The residual architecture itself also contributes: because each layer adds a small increment rather than overwriting the representation, gradients can flow backward without vanishing or exploding. Skip connections provide a "gradient highway," which is the deeper reason why residual networks train more easily than plain deep networks (He et al., 2015).

## The Stickiness Problem

The residual stream's greatest strength — persistence of information — is also its potential weakness. Information written early is never erased, only supplemented, so it can become **sticky**: an early representation appropriate for a low-level task may persist and interfere with higher-level processing.

This motivates architectural experiments that introduce **brand-new representation spaces** at certain depths — forcing the network to re-encode its understanding from scratch rather than incrementally refining a single stream. This is one of the intuitions behind mixture-of-experts routing and proposals for deeper architectural modularity.

## Linear Representations: The Expected Geometry of a Linear Machine

There is a deeper reason to expect that representations inside a transformer are approximately linear: both attention and MLP layers process information through predominantly linear operations. If linear operations dominate the forward pass, the geometry of the representation space that minimizes prediction error should itself be *approximately linear*.

This is exactly what we observe. The most famous example remains word2vec's discovery:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

Gender and royalty are encoded as approximately linear directions in embedding space. Dozens of such results have been found — verb tense, country-capital pairs, comparative adjectives. Linear representations are also computationally desirable: a linear classifier (which is all that a key-query dot product or an MLP's first layer implements) can perfectly separate linearly encoded features without complex decoding.

## Non-Linear Representations: The Frontier

Does linearity have limits? Almost certainly. For sufficiently complex tasks requiring nested logical structure or fine-grained disambiguation, a purely linear geometry may be insufficient.

Anthropic's work on **superposition** (*"Toy Models of Superposition,"* Elhage et al. 2022) provides a compelling hint. The hypothesis is that networks represent *more features than they have dimensions* by storing features as nearly-orthogonal directions in a lower-dimensional space — already a departure from a clean linear regime, as features interfere with each other in ways that resemble aggregate non-linearity.

Deeper models may implement **non-linear representations** by building them up gradually: one layer encodes a linear approximation, subsequent layers correct the error terms, and the composition of many near-linear stages implements a genuinely curved geometry. This is analogous to how piecewise-linear functions approximate smooth curves. Whether this is efficient remains debated, but for tasks at the frontier of model capability, the additional expressive power may be worth the cost.

## A Unified Picture

All information flows through the **residual stream** — a shared vector for each token accumulating contributions from every layer. **Attention** exchanges information across tokens via context-dependent linear combinations and writes small updates back. **MLP layers** perform soft key-value lookups into weight-stored knowledge and add retrieved information back. **Layer normalization** keeps the stream on a bounded manifold, stabilizing optimization.

The overall process is progressive refinement: a rough sketch at the input becomes a detailed, contextually informed, knowledge-enriched vector at the output. The loss landscape is smooth because residual connections provide gradient highways; representations are approximately linear because linear operations dominate; and deeper models can, in principle, compose these linear stages into richer, non-linear geometries.

The residual stream is the transformer's memory, workspace, and communication bus simultaneously — a single elegant data structure that makes possible the remarkable breadth of what these models can do.

---

**References**

- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*, Anthropic
- Geva et al. (2020), *Transformer Feed-Forward Layers Are Key-Value Memories*
- Elhage et al. (2022), *Toy Models of Superposition*, Anthropic
- He et al. (2015), *Deep Residual Learning for Image Recognition*
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*