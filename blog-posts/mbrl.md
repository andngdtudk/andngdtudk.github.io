<!-- image: https://andngdtudk.github.io/images/go.jpg -->

# Model-Based RL and Planning: Learning to Simulate and Search

*Munich*, 17th April 2026

In our previous post, we explored how reinforcement learning enables agents to learn optimal behavior through trial and error, without requiring explicit models of the environment. Model-free methods like Q-learning and policy gradients have achieved remarkable successes, from playing Atari games to controlling robots. But they share a fundamental limitation: they are **sample inefficient**. Learning purely from experience requires enormous amounts of interaction with the environment — often millions of trials to master tasks that humans can learn from just a few examples.

The key to human-like sample efficiency lies in **mental simulation**. When you consider crossing a busy street, you don't need to actually step into traffic multiple times to learn it's dangerous. You imagine possible futures — "if I step out now, that car will hit me" — using your internal model of how the world works. This is the essence of **model-based reinforcement learning**: learning a model of the environment's dynamics and using it to simulate and plan before acting.

Model-based RL represents a synthesis of everything we've covered: it uses system identification from control theory to learn dynamics, employs optimization methods for trajectory planning, leverages the MDP framework for value estimation, and when combined with tree search, achieves superhuman performance in complex domains like chess and Go. This post explores the mathematical foundations and algorithms that make model-based RL so powerful.

## The Model-Based RL Paradigm

The core idea is elegantly simple: if we have a model of how the environment responds to our actions, we can use it to simulate experience and plan optimal behavior.

### The Architecture

A model-based RL agent consists of:

1. **Model learning**: Learn dynamics $\hat{P}(s'|s,a)$ and rewards $\hat{r}(s,a)$ from experience
2. **Planning**: Use the model to simulate trajectories and evaluate policies
3. **Policy learning**: Update policy based on simulated experience
4. **Real interaction**: Act in environment, gather data, update model

This creates a virtuous cycle: better models enable better planning, which generates better data for model improvement.

### Mathematical Formulation

Given a dataset $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^N$ of transitions, learn:

**Dynamics model**: 
$$\hat{P}_{\boldsymbol{\phi}}(s'|s,a) \quad \text{or} \quad s' = f_{\boldsymbol{\phi}}(s,a) + \boldsymbol{\epsilon}$$

**Reward model**:
$$\hat{r}_{\boldsymbol{\psi}}(s,a)$$

The learned model $M = (\hat{P}_{\boldsymbol{\phi}}, \hat{r}_{\boldsymbol{\psi}})$ defines an approximate MDP that we can use for planning.

### Why Model-Based Methods?

**Sample efficiency**: One real transition can be used to generate many simulated transitions. If planning uses $K$ simulated steps per real transition, we effectively multiply our data by factor $K$.

**Transfer**: A good model can generalize to new tasks or reward functions without additional real experience.

**Interpretability**: The learned model provides insight into the agent's understanding of the environment.

**Safety**: Dangerous scenarios can be explored in simulation rather than reality.

**Challenges**: Model errors compound over long horizons; the "model exploitation" problem where the agent finds unrealistic strategies that exploit model inaccuracies.

## Learning Dynamics Models

The first challenge is learning an accurate model from data.

### Deterministic Dynamics

For deterministic environments, we want:
$$f_{\boldsymbol{\phi}}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$$

such that $s_{t+1} = f_{\boldsymbol{\phi}}(s_t, a_t)$.

**Loss function** (supervised learning):
$$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[\|s' - f_{\boldsymbol{\phi}}(s,a)\|^2\right]$$

**Neural network parameterization**: Use deep networks to approximate complex dynamics:
$$f_{\boldsymbol{\phi}}(s,a) = \text{NN}_{\boldsymbol{\phi}}([s; a])$$

where $[s; a]$ denotes concatenation.

### Stochastic Dynamics

Real environments are often stochastic. Model the distribution over next states:

**Gaussian process models**:
$$s' \sim \mathcal{N}(\mu_{\boldsymbol{\phi}}(s,a), \Sigma_{\boldsymbol{\phi}}(s,a))$$

**Mixture density networks**: Output parameters of a mixture distribution:
$$P(s'|s,a) = \sum_{k=1}^K \pi_k(s,a) \mathcal{N}(s' | \mu_k(s,a), \Sigma_k(s,a))$$

**Probabilistic neural networks**: Use dropout or ensembles to capture epistemic uncertainty.

### Model Uncertainty

A critical consideration is distinguishing:
- **Aleatoric uncertainty**: Inherent stochasticity in the environment
- **Epistemic uncertainty**: Uncertainty due to limited data

**Ensemble models**: Train $M$ models $\{f_{\boldsymbol{\phi}_i}\}_{i=1}^M$ with different initializations:
$$\mu(s,a) = \frac{1}{M}\sum_{i=1}^M f_{\boldsymbol{\phi}_i}(s,a)$$
$$\sigma^2(s,a) = \frac{1}{M}\sum_{i=1}^M \|f_{\boldsymbol{\phi}_i}(s,a) - \mu(s,a)\|^2$$

High variance indicates regions where the model is uncertain.

**Bayesian neural networks**: Maintain distributions over weights:
$$p(\boldsymbol{\phi} | \mathcal{D}) \propto p(\mathcal{D} | \boldsymbol{\phi}) p(\boldsymbol{\phi})$$

Sample multiple weight configurations to quantify uncertainty.

### Latent Space Models

For high-dimensional observations (images), learn in latent space:

**World models**: Learn encoder-decoder + dynamics:
$$z_t = \text{Encoder}(o_t)$$
$$z_{t+1} = f_{\boldsymbol{\phi}}(z_t, a_t)$$
$$\hat{o}_t = \text{Decoder}(z_t)$$

Train end-to-end with reconstruction loss + dynamics prediction loss.

**PlaNet/Dreamer**: Use recurrent state-space models:
$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$$
$$z_t \sim q(z_t | h_t, o_t)$$

where $h_t$ is a deterministic hidden state and $z_t$ is a stochastic latent state.

## Planning with Learned Models

Once we have a model, we can use it for planning — computing optimal action sequences.

### Model Predictive Control (MPC)

At each time step $t$, solve an optimization problem:

$$\max_{a_t, \ldots, a_{t+H-1}} \sum_{k=0}^{H-1} r(s_{t+k}, a_{t+k})$$

subject to:
$$s_{t+k+1} = f_{\boldsymbol{\phi}}(s_{t+k}, a_{t+k})$$
$$s_t = \text{current state}$$

Execute $a_t^*$, observe next state, and repeat (receding horizon).

**Random shooting**: Sample $N$ action sequences, evaluate using model, choose best.

**Cross-Entropy Method (CEM)**: Iteratively refine distribution over action sequences:
1. Sample $N$ sequences from $\pi(a_{t:t+H-1})$
2. Evaluate returns using model
3. Keep top $K$ sequences (elites)
4. Fit new distribution to elites
5. Repeat until convergence

**Gradient-based optimization**: If model is differentiable, use gradient descent:
$$\boldsymbol{a}^* = \arg\max_{\boldsymbol{a}} \sum_{k=0}^{H-1} r(s_k, a_k)$$

where $s_{k+1} = f_{\boldsymbol{\phi}}(s_k, a_k)$.

Compute gradients via backpropagation through time.

### Dyna Architecture

**Dyna** integrates model-free learning with model-based planning.

**Algorithm**:
1. Take action $a$ in environment, observe $(s, a, r, s')$
2. Update model: $\hat{P}(s'|s,a)$, $\hat{r}(s,a)$
3. Update value function (model-free): $Q(s,a) \leftarrow \text{Q-learning update}$
4. **Planning step**: For $n$ iterations:
   - Sample previously observed $(s, a)$
   - Simulate: $r, s' \sim \hat{P}(\cdot|s,a)$
   - Update $Q(s,a)$ from simulated experience
5. Repeat

**Key insight**: Real experience updates both model and values; model generates simulated experience for additional value updates.

### MBPO: Model-Based Policy Optimization

**Janner et al., 2019** combines model-based rollouts with model-free policy optimization:

1. Collect data using current policy
2. Train ensemble of dynamics models
3. Generate rollouts from real states using model
4. Add model rollouts to replay buffer
5. Update policy using SAC (or other off-policy algorithm)

**Branched rollouts**: Start from real states, rollout for $k$ steps using model. This limits error accumulation.

**Short horizons**: Use $k = 1$ to $5$ steps to avoid compounding errors. Even short rollouts significantly improve sample efficiency.

## Monte Carlo Tree Search (MCTS)

For discrete action spaces and perfect models (e.g., board games), **Monte Carlo Tree Search** combines tree search with Monte Carlo evaluation.

### The MCTS Algorithm

MCTS builds a search tree incrementally through four phases:

**1. Selection**: Start at root, traverse tree using selection policy until reaching a leaf node. Common selection: **Upper Confidence Bound for Trees (UCT)**:

$$a^* = \arg\max_a \left[Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}\right]$$

where:
- $Q(s,a)$ is the average value of action $a$ from state $s$
- $N(s)$ is visit count of state $s$
- $N(s,a)$ is visit count of state-action pair
- $c$ is exploration constant

This balances exploitation (high $Q$-values) with exploration (low visit counts).

**2. Expansion**: Add one or more child nodes to the tree.

**3. Simulation (Rollout)**: From the new node, simulate to terminal state using a rollout policy (often random or simple heuristic).

**4. Backpropagation**: Propagate the outcome back up the tree, updating statistics:
$$N(s,a) \leftarrow N(s,a) + 1$$
$$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}[G - Q(s,a)]$$

where $G$ is the return from the simulation.

After sufficient iterations, select action:
$$a^* = \arg\max_a N(s,a) \quad \text{or} \quad a^* = \arg\max_a Q(s,a)$$

### Why MCTS Works

**Asymptotic optimality**: As the number of simulations $n \to \infty$, MCTS converges to the minimax-optimal action.

**Anytime algorithm**: Can be stopped at any time and return best action found so far.

**Selective search**: Focuses computational effort on promising branches, avoiding exhaustive search.

**No domain knowledge required**: Works with just forward model and terminal value function.

### UCT Analysis

The UCT formula is derived from UCB for multi-armed bandits applied to tree nodes.

**Regret bound**: After $n$ simulations, the regret (suboptimality) is:
$$\Delta_n = O\left(\frac{\ln n}{n}\right)$$

This logarithmic regret is optimal for the bandit problem and extends to tree search under certain conditions.

## AlphaGo and AlphaZero: Deep Learning Meets MCTS

AlphaGo combined deep neural networks with MCTS to achieve superhuman Go performance.

### AlphaGo Architecture

Three neural networks:

**Policy network** $p_{\boldsymbol{\theta}}(a|s)$: Predicts human expert moves
**Value network** $v_{\boldsymbol{\phi}}(s)$: Estimates probability of winning from state $s$
**Fast rollout policy** $p_{\pi}(a|s)$: Lightweight policy for quick simulations

**Training**:
1. Train policy network via supervised learning on expert games
2. Improve via policy gradient reinforcement learning (self-play)
3. Train value network to predict game outcomes from self-play positions

**MCTS with neural networks**: Modified MCTS using:
- Policy network to guide selection and expansion
- Value network to evaluate leaf nodes (replacing rollouts)

### AlphaZero: Mastering Games Through Self-Play

**AlphaZero** (Silver et al., 2017) eliminated human knowledge entirely:

**Single neural network**: Outputs both policy and value:
$$p, v = f_{\boldsymbol{\theta}}(s)$$

where $p \in \mathbb{R}^{|\mathcal{A}|}$ is a policy vector and $v \in \mathbb{R}$ is a value estimate.

**Training loop**:
1. **Self-play**: Generate games using MCTS guided by $f_{\boldsymbol{\theta}}$
2. **Training data**: Each position $(s, \boldsymbol{\pi}, z)$ where:
   - $s$ is the board position
   - $\boldsymbol{\pi}$ is the MCTS policy (visit counts)
   - $z \in \{-1, +1\}$ is the game outcome
3. **Network update**: Minimize loss:
$$\mathcal{L}(\boldsymbol{\theta}) = (z - v)^2 - \boldsymbol{\pi}^T \log \mathbf{p} + c\|\boldsymbol{\theta}\|^2$$

The first term is value error, the second is policy loss (cross-entropy), the third is regularization.

### MCTS in AlphaZero

**Selection**: Use PUCT (Predictor + UCT):
$$a^* = \arg\max_a \left[Q(s,a) + c \cdot p_a \frac{\sqrt{N(s)}}{1 + N(s,a)}\right]$$

where $p_a$ is the prior probability from the policy network.

**Expansion**: Add all actions to tree, initialize with prior $p$.

**Evaluation**: Use value network instead of rollouts:
$$v = f_{\boldsymbol{\theta}}(s)$$

**Backpropagation**: Update $Q(s,a)$ as before using $v$.

### Why AlphaZero Works

**Neural networks as heuristics**: $p$ guides search toward good moves; $v$ evaluates positions without rollouts.

**Self-play curriculum**: Training on own games provides progressively harder opponents.

**MCTS as policy improvement**: MCTS policy $\boldsymbol{\pi}$ is stronger than raw network policy $\mathbf{p}$; training network to match $\boldsymbol{\pi}$ improves it.

**Value equivalence**: After sufficient training, $v \approx V^{\pi_{\text{MCTS}}}$, making search and evaluation consistent.

### Results

AlphaZero achieved:
- **Chess**: Defeated Stockfish 8 (world champion engine) after 4 hours of training
- **Shogi**: Defeated champion program Elmo after 2 hours
- **Go**: Defeated AlphaGo Lee (which beat world champion Lee Sedol) after 8 hours

Starting only with game rules — no human knowledge, opening books, or endgame tables.

## MuZero: Model-Based RL Without Knowing the Rules

**MuZero** (Schrittwieser et al., 2020) extends AlphaZero to domains where rules are unknown.

### Key Insight

We don't need to predict actual observations $o_{t+1}$. We only need to predict quantities relevant for planning: rewards and values.

**Latent dynamics model**: Learn:
$$h_0 = h_{\boldsymbol{\theta}}(o_1, \ldots, o_t) \quad \text{(representation)}$$
$$r_k, h_k = g_{\boldsymbol{\theta}}(h_{k-1}, a_k) \quad \text{(dynamics)}$$
$$p_k, v_k = f_{\boldsymbol{\theta}}(h_k) \quad \text{(prediction)}$$

where $h_k$ is a latent state, not required to match true environment state.

**Planning**: Run MCTS in latent space using learned dynamics $g_{\boldsymbol{\theta}}$.

**Training**: From self-play trajectories, minimize:
$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{k=0}^{K} \left[(z_{t+k} - v_k)^2 - \boldsymbol{\pi}_{t+k}^T \log \mathbf{p}_k + (u_{t+k} - r_k)^2\right]$$

where $u_{t+k}$ are actual observed rewards and $z_{t+k}$ are $n$-step bootstrapped returns.

**Unrolling**: Train by unrolling $K$ steps in latent space, comparing predictions to real outcomes.

### Results

MuZero matched AlphaZero on Go, chess, shogi, and achieved state-of-the-art on Atari — learning both rules and strategy simultaneously.

## World Models: Learning to Dream

**Ha & Schmidhuber, 2018** trained agents entirely inside learned world models.

### Architecture

**Vision model (V)**: Variational autoencoder compressing observations:
$$z_t = \text{Encode}(o_t), \quad \hat{o}_t = \text{Decode}(z_t)$$

**Memory model (M)**: RNN predicting next latent state:
$$P(z_{t+1} | a_t, z_t, h_t)$$

where $h_t$ is hidden state.

**Controller (C)**: Linear policy mapping $(z_t, h_t)$ to actions:
$$a_t = W_c [z_t; h_t]$$

### Training Procedure

1. **Collect data**: Random policy generates observations
2. **Train V**: VAE on observations
3. **Train M**: RNN on latent sequences with actions
4. **Train C**: Evolution strategies in dream (simulate using M)

Agent is trained entirely in imagination after initial random data collection!

### Results

Solved CarRacing environment without seeing real frames during training — only dreamed simulations.

## Dyna-2: Combining Real and Simulated Experience

**Silver et al., 2008** proposed learning two value functions:

**Long-term memory**: Learn permanent features $V_{\text{perm}}(s; \mathbf{w})$ from real experience.

**Short-term memory**: Learn transient features $V_{\text{trans}}(s; \boldsymbol{\theta})$ from simulated experience with learned model.

Combined value:
$$V(s) = V_{\text{perm}}(s; \mathbf{w}) + V_{\text{trans}}(s; \boldsymbol{\theta})$$

**Key idea**: Separate what's learned from reality (reliable, but sample-expensive) from what's learned from simulation (sample-efficient, but potentially inaccurate).

Reset $\boldsymbol{\theta}$ periodically to avoid model exploitation.

## Model Errors and Robust Planning

A fundamental challenge: model errors compound over planning horizons.

### Pessimistic Planning

Use model uncertainty to be conservative:

**Lower confidence bound**:
$$V_{\text{LCB}}(s) = \mu_{\text{model}}(s) - \beta \sigma_{\text{model}}(s)$$

Plan using pessimistic values to avoid optimistic exploitation of model errors.

**MOREL** (Kidambi et al., 2020): Construct pessimistic MDP with penalties on uncertain transitions.

### Model Ensembles

Train ensemble $\{M_i\}_{i=1}^N$ and use:
- **Mean prediction**: $s' = \frac{1}{N}\sum_i f_i(s,a)$
- **Disagreement**: $\sigma^2 = \frac{1}{N}\sum_i \|f_i(s,a) - \bar{f}(s,a)\|^2$

High disagreement indicates unreliable regions.

**MBPO strategy**: Sample model from ensemble for each rollout.

### Short-Horizon Rollouts

Limit planning depth to reduce error accumulation:
- Horizon 1-5 steps typical in MBPO
- Trade off: shorter horizons need more accurate value functions

### Model-Free Fallback

Hybrid approaches:
- Use model for exploration and initial learning
- Switch to model-free for final policy refinement
- Maintain both and interpolate: $\pi = \alpha \pi_{\text{model}} + (1-\alpha)\pi_{\text{free}}$

## Theoretical Analysis

### Sample Complexity

**Model-based bound**: With $\epsilon$-accurate model:
$$\tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^2(1-\gamma)^3}\right)$$

transitions to learn model, then planning is free.

**Model-free bound**: 
$$\tilde{O}\left(\frac{|\mathcal{S}|^2|\mathcal{A}|}{\epsilon^2(1-\gamma)^4}\right)$$

Factor $|\mathcal{S}|$ worse! Model-based can be exponentially more sample-efficient.

### When Does Model-Based Help?

**Advantages**:
- Sample efficiency (amortize model learning across tasks)
- Transfer (model applies to new reward functions)
- Safety (test dangerous scenarios in simulation)

**Disadvantages**:
- Model errors can lead to poor policies
- Computational cost of planning
- Difficulty modeling complex environments (e.g., humans, physics)

**Rule of thumb**: Model-based when:
- Sample collection is expensive
- Model is easier to learn than policy directly
- Planning horizon is short to moderate
- Environment dynamics are smooth and structured

## Applications to Mobility and Robotics

### Autonomous Vehicle Motion Planning

**Model**: Vehicle kinematics/dynamics
$$\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \\ \dot{v} \end{bmatrix} = \begin{bmatrix} v\cos\theta \\ v\sin\theta \\ v\tan\delta/L \\ a \end{bmatrix}$$

where $\delta$ is steering angle, $L$ is wheelbase.

**Planning**: MPC with learned model for:
- Obstacle trajectories (other vehicles)
- Tire-road friction
- Actuator dynamics

**Approach**:
- Learn residual dynamics: $s_{t+1} = f_{\text{physics}}(s_t, a_t) + f_{\text{learned}}(s_t, a_t)$
- Physics model provides structure, neural network captures complexities
- Plan trajectories via optimization or sampling

### Robot Manipulation

**Model**: Object dynamics under contact
- Difficult to model analytically (friction, deformation, etc.)
- Learn from vision using forward models

**One-shot imitation**: Given single demonstration, use model to plan variations:
1. Observe demonstration trajectory
2. Learn local dynamics model
3. Plan new trajectory to similar goal using MPC

### Traffic Flow Prediction

**Model**: Macroscopic traffic dynamics
$$\frac{\partial \rho}{\partial t} + \frac{\partial(\rho v)}{\partial x} = 0$$

where $\rho$ is density, $v = V(\rho)$ is speed-density relationship.

**Learning**: Neural network $V_{\boldsymbol{\theta}}(\rho, x, t)$ learns from data.

**Planning**: Signal control via MPC using learned dynamics.

### Fleet Repositioning

**Model**: Demand distribution $d(x, t)$ and trip patterns $T(x \to y)$

**Learning**: Historical data → neural network predicting spatiotemporal patterns

**Planning**: Where to reposition idle vehicles to minimize wait times

**Approach**:
- Model-based: Simulate demand, optimize repositioning
- Faster than model-free RL in dynamic environments
- Updates as demand patterns change

### Energy System Control

**Building HVAC**: 
- Model: Thermal dynamics, occupancy patterns, weather
- Planning: MPC to minimize energy cost while maintaining comfort

**EV charging**:
- Model: Battery dynamics, price forecasts, driving patterns
- Planning: When to charge to minimize cost

**Model benefits**: 
- Physical models provide structure
- Data-driven learning captures building-specific effects
- Uncertainty quantification for robust planning

## The Spectrum of Model-Based Methods

Model-based RL encompasses a spectrum:

**Explicit models** → **Implicit models**
- Known physics (LQR, MPC) → Learned world models → Latent dynamics (MuZero)

**Full model** → **Value-equivalent model**
- Predict observations → Predict only reward/value → Learn implicit planning (decision transformers)

**Long rollouts** → **Short rollouts**
- Plan to horizon → 5-step rollouts (MBPO) → 1-step Dyna → Pure model-free

**Deterministic** → **Stochastic**
- Deterministic dynamics → Gaussian → Ensembles → Full distribution

The right choice depends on:
- Domain characteristics (stochastic? high-dim?)
- Sample budget (expensive data favors model-based)
- Computational budget (planning is expensive)
- Robustness requirements (model errors acceptable?)

## Looking Ahead: Transformers and Sequence Models

We've now covered the core frameworks of intelligent decision-making:
- Systems evolve according to dynamics
- Control theory guides single-agent optimization
- Game theory handles multi-agent interaction
- MDPs formalize sequential decisions under uncertainty
- Model-based methods enable sample-efficient learning through planning

In our next post, we'll explore how **Transformers and sequence models** are revolutionizing RL and planning. We'll see how:
- Attention mechanisms enable reasoning over long contexts
- Decision Transformers reframe RL as sequence prediction
- Trajectory optimization in latent spaces enables efficient planning
- Foundation models pretrained on massive data enable few-shot adaptation
- Transformers unify perception, planning, and control

This represents the frontier where classical methods meet modern deep learning — where the mathematical foundations we've built meet the empirical power of large-scale learning. The result is AI systems that can plan, adapt, and generalize in ways that approach human-like flexibility.

---

**References**

**Sutton, R. S.** (1991). Dyna, an integrated architecture for learning, planning, and reacting. *ACM SIGART Bulletin*, 2(4), 160–163.

**Browne, C. B., et al.** (2012). A survey of Monte Carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(1), 1–43.

**Silver, D., et al.** (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484–489.

**Silver, D., et al.** (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. *arXiv preprint arXiv:1712.01815*.

**Schrittwieser, J., et al.** (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature*, 588(7839), 604–609.

**Ha, D., & Schmidhuber, J.** (2018). World models. *arXiv preprint arXiv:1803.10122*.

**Janner, M., et al.** (2019). When to trust your model: Model-based policy optimization. *Advances in Neural Information Processing Systems*, 32.

**Chua, K., et al.** (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. *Advances in Neural Information Processing Systems*, 31.

**Kidambi, R., et al.** (2020). MOReL: Model-based offline reinforcement learning. *Advances in Neural Information Processing Systems*, 33.
