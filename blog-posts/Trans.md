<!-- image: https://andngdtudk.github.io/images/nlp.PNG -->

# Transformers and Sequence Models: The Future of Decision-Making AI

*Munich*, 14th May 2026

In 2017, the paper "Attention Is All You Need" introduced the Transformer architecture, sparking a revolution in natural language processing. Models like GPT and BERT demonstrated unprecedented abilities to understand and generate text by learning from massive datasets. By 2020, language models had become so capable that they could write coherent essays, answer questions, and even write code — all by predicting the next token in a sequence.

But here's a profound insight: **decision-making is sequence modeling**. When an agent interacts with an environment, it generates a sequence: state, action, reward, state, action, reward... Just as language models predict the next word given context, decision-making agents predict the next action given history. This connection is not merely analogous — it's fundamental. And it suggests that the same architectures and training methods that revolutionized NLP might also transform reinforcement learning and planning.

This is exactly what's happening. Transformers are now being applied to sequential decision-making with remarkable success. Decision Transformers reframe RL as conditional sequence modeling. Trajectory Transformers learn world models in sequence space. Gato demonstrates a single model that can play Atari, caption images, stack blocks with a robot arm, and chat — all by treating everything as a sequence prediction problem. This post explores how Transformers and sequence models are reshaping the landscape of decision-making AI.

## The Transformer Architecture: A Brief Review

Before diving into applications, let's review the key components of Transformers.

### Self-Attention Mechanism

The core innovation is **self-attention**: a mechanism that computes contextualized representations by attending to all positions in a sequence.

Given a sequence of tokens $\mathbf{x}_1, \ldots, \mathbf{x}_T$, compute **queries**, **keys**, and **values**:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

where $\mathbf{X} = [\mathbf{x}_1; \ldots; \mathbf{x}_T] \in \mathbb{R}^{T \times d}$ is the input matrix.

**Attention scores**: Compute similarity between queries and keys:
$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

where $d_k$ is the key dimension. Entry $A_{ij}$ represents how much position $i$ attends to position $j$.

**Output**: Weighted combination of values:
$$\mathbf{Z} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{T \times d_v}$$

**Multi-head attention**: Run multiple attention mechanisms in parallel:
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$.

### Transformer Block

A standard Transformer block consists of:
1. **Multi-head self-attention** with residual connection and layer normalization
2. **Feed-forward network** (two linear layers with non-linearity) with residual and normalization

$$\mathbf{Z}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}))$$
$$\mathbf{Z}'' = \text{LayerNorm}(\mathbf{Z}' + \text{FFN}(\mathbf{Z}'))$$

### Positional Encoding

Since attention is permutation-invariant, we add **positional encodings**:
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

Or learn absolute/relative positional embeddings.

### Why Transformers Are Powerful

**Long-range dependencies**: Attention connects all positions directly, unlike RNNs that propagate through hidden states.

**Parallelization**: Unlike RNNs, all positions can be processed in parallel during training.

**Flexible receptive field**: Each position can attend to any other position, with learned attention patterns.

**Scalability**: Performance improves with model size and data (scaling laws).

## Sequential Decision-Making as Sequence Modeling

The key insight connecting Transformers to RL: an agent's experience is a sequence.

### The Trajectory as a Sequence

An episode generates a trajectory:
$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots, s_T)$$

We can represent this as a sequence of tokens:
$$[\mathbf{s}_0, \mathbf{a}_0, r_1, \mathbf{s}_1, \mathbf{a}_1, r_2, \ldots]$$

**Language modeling objective**: Predict next token given previous tokens:
$$p(\tau) = \prod_{t=0}^T p(s_t | s_{<t}, a_{<t}, r_{\leq t}) \cdot p(a_t | s_{\leq t}, a_{<t}, r_{\leq t}) \cdot p(r_{t+1} | s_{\leq t}, a_{\leq t})$$

If we can model this distribution accurately, we can:
- **Imitate**: Sample actions from $p(a_t | \text{history})$
- **Plan**: Search over action sequences that maximize expected return
- **Generate**: Simulate trajectories without environment interaction

### From RL to Supervised Learning

Traditional RL maximizes expected return:
$$\max_{\pi} \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T r_t\right]$$

This requires:
- Exploration to discover good behaviors
- Credit assignment across time
- Handling non-stationarity and bootstrapping

Sequence modeling reframes this as:
$$\max_{\theta} \mathbb{E}_{\tau \sim \mathcal{D}}[\log p_{\theta}(\tau)]$$

This is **supervised learning** on trajectory data! No TD errors, no policy gradients, no exploration strategies — just predict the next token.

**Key question**: How do we ensure the model learns good behaviors, not just any behaviors?

## Decision Transformer: RL via Sequence Modeling

**Chen et al., 2021** introduced the Decision Transformer (DT), which treats RL as conditional sequence modeling.

### Architecture

Instead of predicting unconditionally, condition on **returns-to-go**:
$$\hat{R}_t = \sum_{t'=t}^T r_{t'}$$

The sequence becomes:
$$[\hat{R}_0, \mathbf{s}_0, \mathbf{a}_0, \hat{R}_1, \mathbf{s}_1, \mathbf{a}_1, \ldots]$$

**Model**: A Transformer that predicts actions:
$$p(a_t | \hat{R}_t, s_t, \hat{R}_{<t}, s_{<t}, a_{<t})$$

At test time, specify a desired return $\hat{R}_0 = R_{\text{target}}$ and sample actions autoregressively.

### Training

**Dataset**: Collect trajectories using any policy (including suboptimal ones).

**Loss**: Maximize log-likelihood of actions in trajectories:
$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=0}^T \log p_{\theta}(a_t | \hat{R}_{\geq t}, s_{\geq t}, a_{<t})\right]$$

No Q-functions, no advantage estimation, no policy gradients — just language modeling loss!

### Why This Works

**Conditional generation**: By conditioning on return, we learn the relationship between desired outcomes and actions.

**Stitching**: The model can combine good segments from different trajectories. If we have:
- Trajectory 1: Good start, bad ending (low return)
- Trajectory 2: Bad start, good ending (low return)

A conditional model can learn: "When I want high return from this state, take actions like those in the good segments."

This is **trajectory stitching** — combining sub-optimal experiences to construct optimal behavior.

**Credit assignment via attention**: Attention mechanism naturally handles long-range dependencies, assigning credit across time steps.

### Limitations

**Return conditioning may not be enough**: In complex environments, specifying return doesn't uniquely determine behavior.

**No exploration**: Model only learns from observed data. Cannot discover new strategies.

**Distributional shift**: At test time, conditioning on very high returns may query out-of-distribution.

**Assumption of hindsight**: Requires knowing returns-to-go, which assumes knowing the future during training.

### Results

DT matched or exceeded TD-learning methods on offline RL benchmarks (D4RL) with:
- Simpler algorithm (no bootstrapping, no value networks)
- Better long-horizon credit assignment
- More stable training (no overestimation issues)

## Trajectory Transformer: Planning in Sequence Space

**Janner et al., 2021** extended this idea to planning.

### Architecture

Model the entire trajectory distribution:
$$p(\tau) = \prod_{t=0}^T p(s_t | s_{<t}, a_{<t}, r_{<t}) \cdot p(a_t | s_{\leq t}, a_{<t}, r_{<t}) \cdot p(r_t | s_{\leq t}, a_{<t})$$

A single autoregressive Transformer models transitions, actions, and rewards jointly.

### Discretization

For continuous states/actions, discretize into bins:
$$s \in \mathbb{R}^n \to s \in \{1, 2, \ldots, K\}^n$$

Trade-off between resolution and vocabulary size.

### Beam Search Planning

At test time:
1. Start from current state $s_0$
2. Use **beam search** to generate multiple candidate trajectories
3. Score by cumulative reward: $\sum_t r_t$
4. Execute first action from highest-scoring trajectory
5. Repeat (receding horizon)

**Beam search**: Maintain top-$k$ partial sequences at each step:
- Expand each by sampling next tokens
- Keep top-$k$ by score
- Continue until horizon

This is planning via search in trajectory space!

### Why This Is Powerful

**Model-based without explicit dynamics**: The Transformer implicitly learns $p(s_{t+1}|s_t, a_t)$ as part of the sequence model.

**Joint modeling**: Predicting everything together captures correlations between states, actions, and rewards.

**Flexible planning**: Can optimize for any objective by scoring trajectories accordingly.

**Long-horizon reasoning**: Attention enables reasoning over entire trajectories.

### Inpainting for Planning

Alternative planning method: **inpainting**

Given partial trajectory with missing actions:
$$[s_0, \_, \_, s_1, \_, \_, s_2, \ldots]$$

Fill in missing tokens by sampling from:
$$p(a_0, r_1, a_1, r_2, \ldots | s_0, s_1, s_2, \ldots)$$

Constrain to high-return trajectories during sampling (e.g., classifier guidance).

### Results

Trajectory Transformer achieved strong performance on continuous control tasks, demonstrating that:
- Planning in sequence space is viable
- Discretization doesn't lose too much information
- Beam search provides effective trajectory optimization

## Gato: A Generalist Agent

**Reed et al., 2022** trained a single Transformer on diverse tasks simultaneously.

### Multi-Task Sequence Modeling

Gato processes sequences with heterogeneous modalities:
- **Images**: Patch embeddings (like Vision Transformer)
- **Text**: Token embeddings
- **Continuous values**: Discretized into bins
- **Actions**: Discrete or discretized continuous

All inputs → embeddings → concatenated into unified sequence.

**Training data**:
- 604 distinct tasks
- Atari games, robotic control, image captioning, dialogue
- Real robot manipulation (stacking blocks)

**Training objective**: Standard autoregressive language modeling across all tasks.

### Architecture Details

- **1.2 billion parameters**
- Transformer with 24 layers, 2048 hidden dimensions
- Context length: 1024 tokens
- Multi-head attention with 16 heads

### Task Conditioning

How does Gato know which task to perform?

**Implicit conditioning**: Task identity emerges from context. Initial tokens indicate task (e.g., game screen, robot observation), and the model infers appropriate behavior.

**Prompt engineering**: Providing examples in context guides behavior (few-shot learning).

### Results

Gato achieved:
- Strong performance on diverse tasks with single set of weights
- **Positive transfer**: Training on many tasks improved performance on individual tasks compared to task-specific models
- **Emergent capabilities**: Could perform new tasks by conditioning on demonstrations (in-context learning)

**Implications**: 
- Generalist models are feasible
- Scale + diversity → better generalization
- Same architecture works across domains

### Limitations

- Performance on individual tasks below specialist models
- Requires enormous training compute
- Discretization loses some precision for continuous control
- Unclear how to scale to millions of tasks

## Prompting and In-Context Learning for RL

The success of in-context learning in language models (e.g., GPT-3) suggests similar approaches for decision-making.

### Algorithm Distillation

**Laskin et al., 2022** trained Transformers to implement RL algorithms in-context.

**Key idea**: Train on learning histories, not just final trajectories.

**Training data**: Run RL algorithm (e.g., Q-learning) for $N$ episodes. Sequence:
$$[\tau_1, \tau_2, \ldots, \tau_N]$$

where early trajectories have poor performance, later trajectories improve.

**Model**: Transformer predicting actions from entire history:
$$p(a_t^{(i)} | \tau_1, \ldots, \tau_{i-1}, s_{<t}^{(i)}, a_{<t}^{(i)})$$

**At test time**: Feed in new task trajectories. Model learns in-context, improving behavior as it observes more episodes!

**Result**: Model learns to implement exploration, credit assignment, and value learning without explicit RL algorithm.

### Meta-RL with Transformers

**Context-based meta-learning**: Learn from few demonstrations of new task.

**Prompt**: Provide expert trajectories:
$$[\tau_{\text{demo1}}, \tau_{\text{demo2}}, s_{\text{current}}]$$

**Query**: Predict action for current state.

Transformer learns to extract task structure from demonstrations and generalize.

### Conditional Generation for Planning

**Guidance**: Condition on desired outcomes:
- Target goal state: $p(a | s, g)$
- Trajectory return: $p(a | s, R_{\text{target}})$
- Constraints: $p(a | s, \text{constraint})$

**Classifier-free guidance**: Train both conditional and unconditional models, interpolate:
$$\log p(a | s, c) = \log p(a | s) + \lambda \nabla_a \log p(c | a, s)$$

This steers generation toward desired conditions.

## Video Prediction and World Models

Transformers can learn world models by predicting future observations.

### Video Transformer

Extend language modeling to video: predict future frames.

**Tokenization**: 
- Discrete: Vector-quantized VAE (VQ-VAE) converts pixels to discrete tokens
- Continuous: Patch embeddings

**Architecture**: 3D attention over space and time.

**Objective**: 
$$p(\mathbf{o}_{t+1:t+H} | \mathbf{o}_{1:t}, \mathbf{a}_{t:t+H-1})$$

Predict future observation sequences conditioned on past and planned actions.

### Masked World Models

**IRIS (Micheli et al., 2022)**: Learns world model via masked prediction.

**Training**:
1. Encode observations to discrete tokens: $z_t = \text{Encode}(o_t)$
2. Create masked sequence: $[\text{mask}, z_1, \text{mask}, z_2, \ldots]$
3. Transformer predicts masked tokens
4. Train with reconstruction loss

**Planning**: Search in latent space, decode to check if goals reached.

**Advantage**: Masked prediction is more sample-efficient than autoregressive.

### Dreamer-Style Transformers

Combine world models with policy learning:
1. Learn Transformer dynamics: $z_{t+1} = f(z_t, a_t)$
2. Learn reward predictor: $r_t = g(z_t)$
3. Train policy via planning in latent space (model-based RL)

Transformers' long-range modeling improves credit assignment in imagination.

## Transformers for Multi-Agent Systems

Attention mechanisms are natural for multi-agent coordination.

### Graph Neural Networks vs. Transformers

**GNNs**: Message passing on fixed communication graph.
**Transformers**: Fully-connected attention (every agent attends to all others).

Transformers learn which agents to attend to, adapting communication structure.

### Multi-Agent Transformer (MAT)

**Architecture**:
- Encode each agent's observation: $h_i = \text{Enc}(o_i)$
- Multi-head attention across agents: $h_i' = \text{Attn}([h_1, \ldots, h_N])$
- Decode to actions: $a_i = \text{Dec}(h_i')$

**Training**: Centralized training with full observations, decentralized execution with local observations.

**Benefits**:
- Learn communication patterns
- Handle variable number of agents
- Scale to large teams

### Emergent Communication

Attention weights reveal learned communication:
- Which agents pay attention to each other
- Information flow patterns
- Role specialization

Visualization shows emergent coordination strategies.

## Theoretical Perspectives

### Transformers as In-Context Optimizers

**Akyürek et al., 2022** showed Transformers can implement gradient descent in-context.

Given sequence $[(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n), x_{n+1}]$, Transformer can:
1. Fit linear model to examples
2. Predict $y_{n+1}$ for $x_{n+1}$

This implements one step of gradient descent on squared loss!

**Extension to RL**: Transformers can implement RL algorithms (Q-learning, policy gradients) in-context by processing trajectory sequences.

### Universal Approximation

**Yun et al., 2020** proved Transformers can approximate any sequence-to-sequence function.

For RL: Can represent any policy $\pi(a | h)$ where $h$ is history.

**Practical limitation**: Requires sufficient capacity and training data.

### Sample Complexity

**Open question**: What is the sample complexity of learning decision-making via sequence modeling vs. traditional RL?

**Empirical observations**:
- Transformers may need more data initially (no inductive bias for credit assignment)
- But generalize better across tasks (transfer learning)
- In-context adaptation can be very sample-efficient

**Trade-off**: Upfront cost of pretraining vs. efficiency of adaptation.

## Scaling Laws for Decision-Making

Inspired by language model scaling laws, researchers study how performance scales with:
- Model size (parameters)
- Dataset size (trajectories)
- Compute (FLOPs)

### Empirical Observations

**Power law relationships**:
$$\text{Loss} \propto N^{-\alpha}$$

where $N$ is model size, data size, or compute, and $\alpha \approx 0.05$ to $0.1$.

**Implications**:
- Larger models consistently outperform smaller ones
- More data → better performance (no saturation yet)
- Compute-optimal scaling: Balance model size and data

**Chinchilla principle**: For fixed compute budget, scale model and data proportionally.

### Transfer Scaling

Performance on downstream tasks improves with pretraining scale:
- Larger pretrained models require fewer fine-tuning samples
- In-context learning improves with scale
- Emergent capabilities appear at certain scales

**Vision**: Train foundation models on diverse decision-making data, adapt to new tasks with minimal samples.

## Challenges and Open Problems

### Long Sequences and Computational Cost

**Problem**: Attention is $O(T^2)$ in sequence length $T$.

**Solutions**:
- Sparse attention patterns (local windows, strided attention)
- Linear attention approximations
- Hierarchical architectures (compress old context)

**Techniques**:
- **Performer**: Linear attention via random features
- **Linformer**: Low-rank approximation of attention matrix
- **Reformer**: Locality-sensitive hashing

### Continuous vs. Discrete

**Discretization challenges**:
- Loss of precision in continuous control
- Large vocabularies for high-dimensional spaces
- Difficulty representing smooth trajectories

**Alternatives**:
- Continuous Transformers (embed without discretization)
- Hybrid: Discrete latents, continuous observations
- Diffusion models (covered in future posts)

### Exploration

Sequence models trained on fixed data don't explore.

**Approaches**:
- **Optimistic planning**: Add exploration bonuses to predicted returns
- **Ensemble disagreement**: Use model uncertainty to guide exploration
- **Active data collection**: Iteratively collect data in uncertain regions

### Causal Structure

Standard attention attends to all positions, but causality flows forward in time.

**Causal masking**: Mask future positions in attention.

**Temporal abstraction**: Hierarchical Transformers with different timescales.

### Safety and Robustness

**Distribution shift**: Model trained on dataset may fail on out-of-distribution states.

**Adversarial robustness**: Attention mechanisms can be fooled.

**Solutions**:
- Uncertainty quantification (ensembles, Bayesian)
- Conservative planning (pessimistic on uncertain regions)
- Verification (formal guarantees for critical applications)

## Applications to Mobility and Logistics

### Autonomous Driving with Transformers

**Scene understanding**: Attention over agents (vehicles, pedestrians) for interaction modeling.

**Trajectory prediction**: Predict future trajectories of all agents jointly.

**Planning**: Generate ego vehicle trajectory considering predictions.

**End-to-end**: Single Transformer from sensors to controls.

**Example - Wayformer**:
- Input: Agent histories, map, ego state
- Multi-head attention across agents and map elements
- Output: Joint trajectory distribution for all agents

### Traffic Flow Forecasting

**Spatiotemporal Transformers**:
- Space: Attention over road network nodes
- Time: Attention over historical time steps
- Predict: Future traffic density/flow

**Benefits**: Capture long-range dependencies in space and time.

### Fleet Routing

**Sequence-to-sequence**: Map customer requests to vehicle routes.

**Attention mechanism**: Learn which customers should be served by which vehicles.

**Combinatorial optimization**: Transformers for TSP, VRP:
1. Encode nodes (customers)
2. Autoregressive decoder selects sequence
3. Attention over unvisited nodes

**Results**: Competitive with OR algorithms, generalize to new problem sizes.

### Warehouse Robotics

**Multi-robot coordination**: Attention-based communication.

**Task allocation**: Transformer decides which robot picks which item.

**Path planning**: Generate collision-free paths for all robots jointly.

## The Path to Foundation Models for Decision-Making

The ultimate vision: **foundation models** pretrained on diverse decision-making data, adaptable to new tasks with minimal fine-tuning.

### Data Requirements

**Scale**: Need millions of trajectories across diverse tasks.

**Sources**:
- Simulations (games, physics engines)
- Real-world datasets (robotics, autonomous driving)
- Human demonstrations
- Synthetic (procedurally generated environments)

### Architecture Choices

**Unified representation**: Single tokenization scheme for all modalities.

**Shared trunk**: Common Transformer layers process all tasks.

**Task-specific heads**: Different output heads for different task types.

### Training Paradigms

**Multi-task learning**: Joint training on all tasks.

**Curriculum learning**: Easy → hard tasks.

**Meta-learning**: Learn to learn from few examples.

**Self-supervised pretraining**: Learn representations before task-specific training.

### Adaptation Strategies

**Fine-tuning**: Update all parameters on new task.

**Prompt engineering**: Provide demonstrations in context.

**Parameter-efficient**: Update small subset (adapters, LoRA).

**Zero-shot**: Specify task via description, no additional training.

## Looking Ahead

We've now explored the full stack of modern decision-making AI:
- Classical foundations (systems, control, optimization, game theory)
- Reinforcement learning (MDPs, value functions, policy gradients)
- Model-based methods (planning, search, world models)
- Modern architectures (Transformers, sequence modeling)

In our final posts, we'll explore:
- **Diffusion models for planning**: How generative models enable flexible trajectory optimization
- **Agentic AI and embodied intelligence**: Putting everything together for real-world systems
- **The future of mobility**: How these technologies will transform transportation and logistics

The convergence of classical methods and modern deep learning is creating AI systems that can perceive, reason, plan, and act in increasingly sophisticated ways. Transformers, with their ability to handle long sequences and model complex dependencies, are becoming the universal architecture for decision-making — just as they became for language and vision.

The mathematical foundations we've built throughout this series remain essential. Transformers don't replace optimal control theory or game theory — they provide powerful function approximators that can learn these principles from data. Understanding both the classical foundations and modern methods is key to building the next generation of intelligent systems.

---

**References**

**Vaswani, A., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

**Chen, L., et al.** (2021). Decision Transformer: Reinforcement learning via sequence modeling. *Advances in Neural Information Processing Systems*, 34.

**Janner, M., et al.** (2021). Offline reinforcement learning as one big sequence modeling problem. *Advances in Neural Information Processing Systems*, 34.

**Reed, S., et al.** (2022). A generalist agent. *Transactions on Machine Learning Research*.

**Laskin, M., et al.** (2022). In-context reinforcement learning with algorithm distillation. *arXiv preprint arXiv:2210.14215*.

**Micheli, V., et al.** (2022). Transformers are sample-efficient world models. *arXiv preprint arXiv:2209.00588*.

**Akyürek, E., et al.** (2022). What learning algorithm is in-context learning? Investigations with linear models. *arXiv preprint arXiv:2211.15661*.

**Zheng, Q., et al.** (2023). Online decision Transformer. *Proceedings of ICML 2023*.
