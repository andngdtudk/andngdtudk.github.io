<!-- image: https://andngdtudk.github.io/images/chess.jpg -->


# Markov Decision Processes and Reinforcement Learning: Sequential Decision-Making Under Uncertainty

*Munich*, 17th March 2026

We've journeyed through the mathematical foundations of intelligent systems: how systems evolve over time, how to control them optimally, how to formulate decision-making as optimization, and how multiple agents interact strategically. Now we arrive at a framework that synthesizes all these ideas: **Markov Decision Processes (MDPs)** and **Reinforcement Learning (RL)**.

An MDP is a mathematical model of sequential decision-making where an agent interacts with a stochastic environment over time. At each step, the agent observes a state, chooses an action, receives a reward, and transitions to a new state. The goal is to learn a policy — a mapping from states to actions — that maximizes cumulative reward over time.

This framework is remarkably general. An autonomous vehicle navigating through traffic is solving an MDP where states are positions and velocities, actions are steering and acceleration, and rewards reflect safety and efficiency. A logistics system routing packages is solving an MDP where states are inventory levels and vehicle positions, actions are routing decisions, and rewards reflect costs and service levels. A robot learning to manipulate objects is solving an MDP through trial and error, discovering effective policies from experience.

MDPs unite our previous frameworks: they extend control theory to unknown dynamics, they formalize optimization over sequential decisions, and when multiple agents interact, they generalize to stochastic games. Reinforcement learning provides the algorithms for solving MDPs when we don't have a complete model of the environment.

This post develops the mathematical theory of MDPs and RL, from the Bellman equations to value iteration and policy gradient methods, from model-based to model-free algorithms, from tabular methods to deep neural network approximation.

## The Markov Decision Process Framework

A **Markov Decision Process** is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$:

- **State space** $\mathcal{S}$: The set of possible states $s \in \mathcal{S}$
- **Action space** $\mathcal{A}$: The set of possible actions $a \in \mathcal{A}$ (or $\mathcal{A}(s)$ for state-dependent action sets)
- **Transition dynamics** $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$: The probability of transitioning to state $s'$ from state $s$ under action $a$:
  $$P(s' | s, a) = \mathbb{P}[S_{t+1} = s' | S_t = s, A_t = a]$$
- **Reward function** $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: The immediate reward for taking action $a$ in state $s$
- **Discount factor** $\gamma \in [0, 1)$: Weighs immediate vs. future rewards

### The Markov Property

The key assumption is the **Markov property**: the future is conditionally independent of the past given the present:
$$\mathbb{P}[S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0] = \mathbb{P}[S_{t+1} | S_t, A_t] = P(S_{t+1} | S_t, A_t)$$

This means the current state contains all information needed to predict the future. The state is a **sufficient statistic** for the history.

### Policies

A **policy** $\pi$ defines the agent's behavior — how it chooses actions based on states.

**Deterministic policy**: $\pi: \mathcal{S} \to \mathcal{A}$
$$a = \pi(s)$$

**Stochastic policy**: $\pi: \mathcal{S} \times \mathcal{A} \to [0,1]$
$$a \sim \pi(\cdot | s), \quad \text{where } \pi(a|s) = \mathbb{P}[A_t = a | S_t = s]$$

For continuous action spaces, $\pi(\cdot | s)$ is a probability density.

### Return and Value Functions

The **return** from time $t$ is the discounted cumulative reward:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

The **state-value function** for policy $\pi$ is the expected return starting from state $s$:
$$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\bigg|\, S_t = s\right]$$

The **action-value function** (Q-function) is the expected return starting from state $s$, taking action $a$:
$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\bigg|\, S_t = s, A_t = a\right]$$

These value functions quantify how good it is to be in a state (or take an action) under policy $\pi$.

## The Bellman Equations: Recursive Structure of Value

The fundamental insight is that value functions satisfy recursive relationships called **Bellman equations**.

### Bellman Expectation Equations

For any policy $\pi$:

**State-value function**:
$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^{\pi}(s')\right]$$

Or more compactly using the Q-function:
$$V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^{\pi}(s,a)$$

**Action-value function**:
$$Q^{\pi}(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^{\pi}(s')$$

Or equivalently:
$$Q^{\pi}(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^{\pi}(s',a')$$

These equations express that the value of a state is the expected immediate reward plus the discounted value of the next state.

### Connection to Dynamic Programming

The Bellman equations connect MDPs to the dynamic programming formulation from control theory. The state-value function is analogous to the value function in optimal control:
$$V(s,t) = \min_{u} \left[L(s,u) + V(f(s,u), t+1)\right]$$

But in MDPs, we work with expectations over stochastic transitions and infinite horizons.

### Bellman Optimality Equations

The **optimal value functions** satisfy:

**Optimal state-value function**:
$$V^*(s) = \max_{\pi} V^{\pi}(s) = \max_{a \in \mathcal{A}} Q^*(s,a)$$

**Optimal action-value function**:
$$Q^*(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^*(s')$$

Combining these:
$$V^*(s) = \max_{a \in \mathcal{A}} \left[r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^*(s')\right]$$

$$Q^*(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \max_{a' \in \mathcal{A}} Q^*(s',a')$$

These are **fixed-point equations**. The optimal value functions are fixed points of the **Bellman optimality operator**.

### Optimal Policies

Once we have $V^*$ or $Q^*$, the optimal policy is:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s,a) = \arg\max_{a \in \mathcal{A}} \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

**Key result**: There always exists a deterministic optimal policy $\pi^*$ that achieves $V^*(s)$ for all states $s$.

**Proof sketch**: If a stochastic policy is optimal, then any action in its support must be optimal (otherwise we could improve by putting more weight on better actions). Therefore, we can construct a deterministic policy by selecting any optimal action at each state. □

## Dynamic Programming Algorithms

When we know the MDP model $(P, r)$, we can compute optimal policies using **dynamic programming**.

### Policy Evaluation

Given a policy $\pi$, compute $V^{\pi}$ by solving the Bellman expectation equation:
$$V^{\pi}(s) = \sum_{a} \pi(a|s) \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi}(s')\right]$$

**Iterative policy evaluation**:
1. Initialize $V_0(s)$ arbitrarily for all $s$
2. Repeat until convergence:
   $$V_{k+1}(s) = \sum_{a} \pi(a|s) \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

This converges to $V^{\pi}$ as $k \to \infty$ because the Bellman operator is a contraction mapping:
$$\|T_{\pi} V - T_{\pi} U\|_{\infty} \leq \gamma \|V - U\|_{\infty}$$

### Policy Improvement

Given $V^{\pi}$, we can improve the policy by acting greedily:
$$\pi'(s) = \arg\max_a \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi}(s')\right]$$

**Policy improvement theorem**: $V^{\pi'}(s) \geq V^{\pi}(s)$ for all $s$.

**Proof**: By construction, for all $s$:
$$Q^{\pi}(s, \pi'(s)) \geq Q^{\pi}(s, \pi(s)) = V^{\pi}(s)$$

Expanding recursively:
$$V^{\pi}(s) \leq Q^{\pi}(s, \pi'(s)) = \mathbb{E}[R_{t+1} + \gamma V^{\pi}(S_{t+1}) | S_t = s, A_t = \pi'(s)]$$
$$\leq \mathbb{E}[R_{t+1} + \gamma Q^{\pi}(S_{t+1}, \pi'(S_{t+1})) | S_t = s, A_t = \pi'(s)]$$
$$\leq \cdots \leq V^{\pi'}(s)$$
□

### Policy Iteration

Alternate between evaluation and improvement:
1. **Policy evaluation**: Compute $V^{\pi_k}$
2. **Policy improvement**: $\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s,a)$
3. Repeat until policy converges

**Convergence**: Policy iteration converges to the optimal policy in finitely many iterations for finite MDPs.

### Value Iteration

Directly iterate the Bellman optimality equation:
$$V_{k+1}(s) = \max_a \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

**Convergence**: $V_k \to V^*$ as $k \to \infty$ at a geometric rate.

**Relationship to policy iteration**: Value iteration performs one sweep of policy evaluation at each step, while policy iteration evaluates the policy to convergence.

### Computational Complexity

For a finite MDP with $|\mathcal{S}|$ states and $|\mathcal{A}|$ actions:
- Each iteration of value iteration: $O(|\mathcal{S}|^2 |\mathcal{A}|)$
- Each iteration of policy evaluation: $O(|\mathcal{S}|^2)$ (or $O(|\mathcal{S}|^3)$ if solving linear system directly)
- Policy iteration typically converges in fewer iterations but each iteration is more expensive

Both are polynomial-time but become intractable for large state spaces, motivating approximation methods.

## Model-Free Reinforcement Learning

In many applications, we don't know the transition dynamics $P$ or reward function $r$. **Model-free RL** learns optimal policies directly from experience without building an explicit model.

### Monte Carlo Methods

**Monte Carlo (MC) methods** learn from complete episodes (trajectories) by averaging returns.

**MC policy evaluation**: To estimate $V^{\pi}(s)$, average the returns observed after visiting $s$:
$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

where $G_t$ is the observed return from time $t$ and $\alpha$ is the learning rate.

**First-visit MC**: Average returns only from first visit to $s$ in each episode
**Every-visit MC**: Average returns from every visit to $s$

**MC control**: Use MC policy evaluation + policy improvement:
1. Generate episode using $\pi$
2. For each $(s,a)$ visited, update $Q(s,a)$ toward observed return
3. Improve policy: $\pi(s) = \arg\max_a Q(s,a)$

**Exploration**: Use $\epsilon$-greedy policies:
$$\pi(a|s) = \begin{cases} 1 - \epsilon + \epsilon/|\mathcal{A}| & \text{if } a = \arg\max_{a'} Q(s,a') \\ \epsilon/|\mathcal{A}| & \text{otherwise} \end{cases}$$

### Temporal-Difference Learning

**Temporal-Difference (TD) methods** learn from incomplete episodes by bootstrapping — updating value estimates based on other value estimates.

**TD(0) for policy evaluation**:
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

The **TD error** is:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

This measures the difference between the current estimate $V(S_t)$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$ (one-step lookahead).

**Advantages over MC**:
- Can learn from incomplete sequences (online learning)
- Lower variance (bootstrapping reduces dependence on future randomness)
- Often faster convergence

**Disadvantages**:
- Biased initially (bootstrap from inaccurate estimates)
- Requires Markov property

### SARSA: On-Policy TD Control

**SARSA** (State-Action-Reward-State-Action) learns $Q^{\pi}$ for the behavior policy:

After transition $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

Then improve policy (e.g., $\epsilon$-greedy on $Q$).

SARSA is **on-policy**: it learns the value of the policy being followed.

### Q-Learning: Off-Policy TD Control

**Q-learning** learns $Q^*$ regardless of the policy being followed:

After transition $(S_t, A_t, R_{t+1}, S_{t+1})$:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

The target uses $\max_a Q(S_{t+1}, a)$ instead of $Q(S_{t+1}, A_{t+1})$.

Q-learning is **off-policy**: the behavior policy (which generates experience) can differ from the target policy (which is being learned).

**Convergence**: Under appropriate conditions (sufficient exploration, decreasing learning rates), Q-learning converges to $Q^*$ with probability 1.

### TD($\lambda$): Eligibility Traces

**TD($\lambda$)** unifies MC and TD by considering $n$-step returns:

**n-step TD target**:
$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

**TD($\lambda$) target** combines all $n$-step returns:
$$G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $\lambda \in [0,1]$ controls the weighting. Special cases:
- $\lambda = 0$: TD(0) — one-step TD
- $\lambda = 1$: Monte Carlo — full episode returns

**Eligibility traces** provide an efficient implementation:
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbb{1}[S_t = s]$$

Update all states:
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

This credits states based on both recency and frequency of visits.

## Function Approximation

For large state spaces, tabular methods are infeasible. We approximate value functions using parameterized functions.

### Value Function Approximation

Represent $V(s) \approx \hat{V}(s; \mathbf{w})$ where $\mathbf{w} \in \mathbb{R}^d$ are learnable parameters.

**Linear function approximation**:
$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(s) = \sum_{i=1}^d w_i \phi_i(s)$$

where $\boldsymbol{\phi}(s) \in \mathbb{R}^d$ are feature vectors.

**Non-linear function approximation** (neural networks):
$$\hat{V}(s; \mathbf{w}) = f_{\mathbf{w}}(s)$$

where $f_{\mathbf{w}}$ is a neural network with parameters $\mathbf{w}$.

### Gradient-Based Updates

Minimize the mean-squared value error:
$$J(\mathbf{w}) = \mathbb{E}_{\pi}[(V^{\pi}(S) - \hat{V}(S; \mathbf{w}))^2]$$

**Gradient descent update**:
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w})$$

For a sample $(S_t, G_t)$:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [G_t - \hat{V}(S_t; \mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(S_t; \mathbf{w})$$

For TD learning, replace $G_t$ with TD target $R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w})$.

### Semi-Gradient Methods

When the target itself depends on $\mathbf{w}$ (as in TD learning), we use **semi-gradient methods** that don't differentiate through the target:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w}) - \hat{V}(S_t; \mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(S_t; \mathbf{w})$$

This may not converge to a true minimum of $J(\mathbf{w})$ but often works well in practice.

### Deep Q-Networks (DQN)

**DQN** combines Q-learning with deep neural networks:

$$\hat{Q}(s, a; \boldsymbol{\theta})$$

where $\boldsymbol{\theta}$ are the network parameters.

**Key innovations**:
1. **Experience replay**: Store transitions $(s, a, r, s')$ in replay buffer $\mathcal{D}$ and sample mini-batches for training
2. **Target network**: Use separate target network $\hat{Q}(s, a; \boldsymbol{\theta}^-)$ for computing targets, updated periodically

**Loss function**:
$$L(\boldsymbol{\theta}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} \hat{Q}(s', a'; \boldsymbol{\theta}^-) - \hat{Q}(s, a; \boldsymbol{\theta})\right)^2\right]$$

**Gradient update**:
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha [r + \gamma \max_{a'} \hat{Q}(s', a'; \boldsymbol{\theta}^-) - \hat{Q}(s, a; \boldsymbol{\theta})] \nabla_{\boldsymbol{\theta}} \hat{Q}(s, a; \boldsymbol{\theta})$$

DQN achieved human-level performance on Atari games, demonstrating that deep RL could work in high-dimensional domains.

## Policy Gradient Methods

Instead of learning value functions and deriving policies, **policy gradient methods** directly optimize parameterized policies $\pi_{\boldsymbol{\theta}}(a|s)$.

### Policy Objective

Maximize expected return:
$$J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[G(\tau)] = \mathbb{E}_{s_0} [V^{\pi_{\boldsymbol{\theta}}}(s_0)]$$

where $\tau = (s_0, a_0, r_1, s_1, a_1, \ldots)$ is a trajectory.

### The Policy Gradient Theorem

**Theorem** (Policy Gradient Theorem): The gradient of the objective is:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}\left[\sum_{t=0}^{\infty} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t | s_t) \, Q^{\pi_{\boldsymbol{\theta}}}(s_t, a_t)\right]$$

This remarkable result shows we can estimate the policy gradient from sample trajectories without knowing the dynamics!

**Proof sketch**: Using the likelihood ratio trick:
$$\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[G(\tau)] = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p(\tau | \boldsymbol{\theta}) \, G(\tau)]$$

The trajectory probability is:
$$p(\tau | \boldsymbol{\theta}) = \mu(s_0) \prod_{t=0}^{\infty} \pi_{\boldsymbol{\theta}}(a_t|s_t) P(s_{t+1}|s_t, a_t)$$

Taking the log-derivative eliminates the unknown dynamics terms:
$$\nabla_{\boldsymbol{\theta}} \log p(\tau | \boldsymbol{\theta}) = \sum_{t=0}^{\infty} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t|s_t)$$

Combining with causality (future actions don't affect past rewards), we get the policy gradient theorem. □

### REINFORCE Algorithm

The **REINFORCE** algorithm uses Monte Carlo sampling:

1. Generate trajectory $\tau$ using $\pi_{\boldsymbol{\theta}}$
2. For each time step $t$, compute return $G_t$
3. Update:
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \sum_{t=0}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t | s_t) \, G_t$$

This is an unbiased but high-variance estimator.

### Variance Reduction: Baselines

Subtract a **baseline** $b(s_t)$ from the return:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}}\left[\sum_{t=0}^{\infty} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t | s_t) \, (Q^{\pi_{\boldsymbol{\theta}}}(s_t, a_t) - b(s_t))\right]$$

This doesn't introduce bias (the baseline term integrates to zero) but reduces variance. A common choice is $b(s) = V^{\pi_{\boldsymbol{\theta}}}(s)$, giving:
$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}}\left[\sum_{t=0}^{\infty} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t | s_t) \, A^{\pi_{\boldsymbol{\theta}}}(s_t, a_t)\right]$$

where $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ is the **advantage function** — how much better action $a$ is than average.

### Actor-Critic Methods

**Actor-critic** algorithms learn both a policy (actor) and a value function (critic):

**Actor**: Policy $\pi_{\boldsymbol{\theta}}(a|s)$ with parameters $\boldsymbol{\theta}$
**Critic**: Value function $\hat{V}(s; \mathbf{w})$ (or $\hat{Q}(s,a; \mathbf{w})$) with parameters $\mathbf{w}$

**Update steps**:
1. **Critic update**: TD learning for value function
   $$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w [r + \gamma \hat{V}(s'; \mathbf{w}) - \hat{V}(s; \mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(s; \mathbf{w})$$

2. **Actor update**: Policy gradient with advantage
   $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_{\theta} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s) \, \delta$$
   
   where $\delta = r + \gamma \hat{V}(s'; \mathbf{w}) - \hat{V}(s; \mathbf{w})$ is the TD error.

Actor-critic methods reduce variance (using the critic) while maintaining low bias (updating from single steps).

### Advanced Policy Gradient Methods

**Trust Region Policy Optimization (TRPO)**: Constrains policy updates to stay within a trust region:
$$\max_{\boldsymbol{\theta}} \mathbb{E}[\mathcal{L}(\boldsymbol{\theta})] \quad \text{s.t.} \quad \overline{D}_{KL}(\pi_{\boldsymbol{\theta}_{\text{old}}}, \pi_{\boldsymbol{\theta}}) \leq \delta$$

where $\overline{D}_{KL}$ is the average KL divergence.

**Proximal Policy Optimization (PPO)**: Simplifies TRPO using a clipped objective:
$$\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \mathbb{E}\left[\min\left(\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} A(s,a), \, \text{clip}\left(\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)}, 1-\epsilon, 1+\epsilon\right) A(s,a)\right)\right]$$

PPO is simpler to implement than TRPO and has become a standard baseline for continuous control.

**Soft Actor-Critic (SAC)**: Adds entropy regularization for exploration:
$$J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}}\left[\sum_{t=0}^{\infty} \gamma^t (r(s_t, a_t) + \alpha \mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot | s_t)))\right]$$

where $\mathcal{H}$ is the entropy. This encourages exploration and makes learning more stable.

## Model-Based Reinforcement Learning

**Model-based RL** learns a model of the environment $(P, r)$ and uses it for planning.

### Learning the Model

Given transitions $(s, a, r, s')$, learn:
- **Dynamics model**: $\hat{P}(s'|s,a)$ or $\hat{s}' = f_{\boldsymbol{\phi}}(s, a)$ for deterministic settings
- **Reward model**: $\hat{r}(s,a) = g_{\boldsymbol{\psi}}(s, a)$

For neural network models, minimize prediction error:
$\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}[\|s' - f_{\boldsymbol{\phi}}(s,a)\|^2]$

### Planning with Learned Models

Once we have a model, we can use it for planning:

**Dyna Architecture**: Integrate learning and planning:
1. Act in environment, observe $(s, a, r, s')$
2. Update model from real experience
3. Update value function from real experience (model-free)
4. Generate simulated experience using model
5. Update value function from simulated experience (model-based)

**Model Predictive Control (MPC) with learned models**:
1. At state $s$, use model to predict future trajectories
2. Optimize action sequence via planning (e.g., random shooting, CEM, trajectory optimization)
3. Execute first action, observe next state
4. Repeat (receding horizon)

**AlphaZero approach**: Combine learned model with tree search (covered in future posts)

### Challenges in Model-Based RL

**Model errors compound**: Small prediction errors accumulate over long horizons. A 1% error per step becomes 37% error over 100 steps.

**Distribution shift**: Model trained on observed states may be inaccurate in regions visited by the learned policy.

**Partial observability**: True state may not be fully observed, requiring belief state planning.

**Solutions**:
- Ensemble models to quantify uncertainty
- Model ensembles and pessimistic value estimates
- Short-horizon planning (MPC)
- Hybrid approaches combining model-based and model-free learning

## Partial Observability: POMDPs

In many domains, the agent doesn't observe the full state. **Partially Observable MDPs (POMDPs)** extend MDPs:

A POMDP adds to the MDP tuple:
- **Observation space** $\mathcal{O}$
- **Observation function** $O: \mathcal{S} \times \mathcal{A} \times \mathcal{O} \to [0,1]$ where $O(o|s',a)$ is the probability of observing $o$ after taking action $a$ and reaching state $s'$

The agent must maintain a **belief state** $b(s) = \mathbb{P}[S_t = s | h_t]$ over states given history $h_t = (a_0, o_1, a_1, \ldots, o_t)$.

**Belief update** (Bayesian filtering):
$b'(s') = \eta \cdot O(o|s',a) \sum_{s \in \mathcal{S}} P(s'|s,a) b(s)$

where $\eta$ is a normalization constant.

The belief state itself is a sufficient statistic, and the POMDP can be converted to an MDP over belief space. However, belief space is continuous even for finite state spaces, making exact solution intractable.

**Approaches**:
- **Point-based value iteration**: Sample beliefs and approximate value function
- **Recurrent networks**: Use RNN/LSTM to encode history: $h_t = f(h_{t-1}, o_t, a_{t-1})$
- **Transformer models**: Attention over historical observations

## Exploration vs. Exploitation

A fundamental challenge in RL is the **exploration-exploitation tradeoff**: should the agent exploit its current knowledge or explore to gather new information?

### Multi-Armed Bandits (Stateless RL)

The simplest RL setting: choose among $K$ actions (arms), each with unknown reward distribution.

**Upper Confidence Bound (UCB)**:
$a_t = \arg\max_a \left[\hat{r}_a + c\sqrt{\frac{\log t}{n_a}}\right]$

where $\hat{r}_a$ is the empirical mean reward of arm $a$, $n_a$ is the number of times it's been played, and $c$ is an exploration constant.

UCB achieves $O(\log T)$ regret, which is optimal.

**Thompson Sampling**: Maintain posterior distribution over reward means, sample from posteriors, and choose action with highest sampled value. Often more robust than UCB in practice.

### Exploration Bonuses in MDPs

**Optimism in the face of uncertainty (OFU)**: Add exploration bonuses to encourage visiting less-explored states:
$\tilde{r}(s,a) = r(s,a) + \beta \sqrt{\frac{\log t}{n(s,a)}}$

where $n(s,a)$ is the visit count.

**Count-based exploration**: Augment rewards based on state novelty:
$\tilde{r}(s,a) = r(s,a) + \beta/\sqrt{n(s)}$

For high-dimensional states, use learned density models or pseudo-counts.

**Curiosity-driven exploration**: Train model to predict next state, use prediction error as intrinsic reward:
$r_{\text{intrinsic}} = \|s_{t+1} - \hat{f}(s_t, a_t)\|^2$

Novel states are harder to predict, encouraging exploration.

**Information gain**: Maximize information about the environment model (Bayesian RL approaches).

## Continuous Action Spaces

Many domains have continuous actions (e.g., robot joint torques, vehicle steering angles).

### Policy Parameterization

**Gaussian policies**: For continuous actions $\mathbf{a} \in \mathbb{R}^m$:
$\pi_{\boldsymbol{\theta}}(\mathbf{a}|\mathbf{s}) = \mathcal{N}(\mathbf{a} | \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{s}), \boldsymbol{\Sigma}_{\boldsymbol{\theta}}(\mathbf{s}))$

where neural networks output mean $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{s})$ and covariance $\boldsymbol{\Sigma}_{\boldsymbol{\theta}}(\mathbf{s})$.

**Deterministic Policy Gradient (DPG)**: For deterministic policies $\mathbf{a} = \mu_{\boldsymbol{\theta}}(\mathbf{s})$:
$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{s} \sim \rho^{\mu}}\left[\nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(\mathbf{s}) \nabla_{\mathbf{a}} Q^{\mu}(\mathbf{s}, \mathbf{a})|_{\mathbf{a} = \mu_{\boldsymbol{\theta}}(\mathbf{s})}\right]$

This allows gradient ascent on expected return using the chain rule through the policy and Q-function.

### Deep Deterministic Policy Gradient (DDPG)

Combines DPG with DQN-style experience replay and target networks:

**Actor**: Deterministic policy $\mu_{\boldsymbol{\theta}}(\mathbf{s})$
**Critic**: Q-function $Q_{\mathbf{w}}(\mathbf{s}, \mathbf{a})$

**Critic update** (minimize TD error):
$\mathcal{L}(\mathbf{w}) = \mathbb{E}[(r + \gamma Q_{\mathbf{w}^-}(\mathbf{s}', \mu_{\boldsymbol{\theta}^-}(\mathbf{s}')) - Q_{\mathbf{w}}(\mathbf{s}, \mathbf{a}))^2]$

**Actor update** (maximize Q-value):
$\nabla_{\boldsymbol{\theta}} J = \mathbb{E}[\nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(\mathbf{s}) \nabla_{\mathbf{a}} Q_{\mathbf{w}}(\mathbf{s}, \mathbf{a})|_{\mathbf{a} = \mu_{\boldsymbol{\theta}}(\mathbf{s})}]$

DDPG is effective for continuous control but can be unstable. TD3 (Twin Delayed DDPG) improves stability with:
- Twin Q-networks (take minimum to reduce overestimation)
- Delayed policy updates
- Target policy smoothing

## Hierarchical Reinforcement Learning

Complex tasks benefit from **hierarchical** decomposition.

### Options Framework

An **option** $\omega = (\mathcal{I}_{\omega}, \pi_{\omega}, \beta_{\omega})$ consists of:
- **Initiation set** $\mathcal{I}_{\omega} \subseteq \mathcal{S}$: states where option can start
- **Policy** $\pi_{\omega}: \mathcal{S} \times \mathcal{A} \to [0,1]$: behavior while executing option
- **Termination condition** $\beta_{\omega}: \mathcal{S} \to [0,1]$: probability of terminating

Options extend actions to **temporally extended** behaviors (skills, sub-policies).

**Semi-MDP over options**: Treat options as actions in a higher-level MDP. Value functions and learning algorithms extend naturally.

### Goal-Conditioned RL

Learn a policy $\pi(\mathbf{a}|\mathbf{s}, \mathbf{g})$ conditioned on goal $\mathbf{g}$.

**Universal Value Function Approximators (UVFAs)**:
$Q(\mathbf{s}, \mathbf{a}, \mathbf{g}) \approx \hat{Q}(\mathbf{s}, \mathbf{a}, \mathbf{g}; \boldsymbol{\theta})$

Learn one function for all goals.

**Hindsight Experience Replay (HER)**: When episode fails to reach goal $\mathbf{g}$, replay experience pretending the achieved state was the goal. Provides dense learning signal even with sparse rewards.

## Offline Reinforcement Learning

**Offline RL** (batch RL) learns from a fixed dataset without environment interaction.

**Challenges**:
- **Distribution shift**: Learned policy visits states not in dataset
- **Overestimation**: Q-learning overestimates values on out-of-distribution actions

**Conservative Q-Learning (CQL)**: Regularize Q-function to be pessimistic on unseen actions:
$\min_Q \mathbb{E}_{(s,a) \sim \mathcal{D}}[(Q(s,a) - \mathcal{B}^* Q(s,a))^2] + \alpha \mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim \pi_{\beta}}[Q(s,a)]\right]$

The regularization term penalizes high Q-values on actions not in the behavior policy $\pi_{\beta}$.

**Behavioral cloning with refinement**: Start with supervised learning on dataset, then fine-tune with online RL.

## Multi-Agent Reinforcement Learning

Extending RL to multiple interacting agents connects back to game theory.

### Independent Q-Learning

Each agent learns independently, treating others as part of the environment:
$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha [r_i + \gamma \max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i)]$

**Problem**: Non-stationarity — as others learn, the environment changes, violating Markov property and convergence guarantees.

### Nash Q-Learning

Learn Nash equilibrium policies by incorporating game-theoretic reasoning:
$Q_i(s, \mathbf{a}) \leftarrow Q_i(s, \mathbf{a}) + \alpha [r_i + \gamma V_i^{\text{Nash}}(s') - Q_i(s, \mathbf{a})]$

where $V_i^{\text{Nash}}(s) = u_i(\mathbf{a}^*)$ for Nash equilibrium $\mathbf{a}^* = \arg\max_{\mathbf{a}} u_i(\mathbf{a})$ given $Q$-values.

**Challenges**: Computing Nash equilibrium at each step; multiple equilibria.

### Mean Field Reinforcement Learning

For large populations, approximate individual agents' impact by population mean:
$Q_i(s_i, a_i) \approx Q(s_i, a_i, \bar{m})$

where $\bar{m}$ is the mean field (distribution over states/actions).

Agents optimize against the mean field, which is updated based on agents' policies. This connects to mean field games from our previous post.

### Communication and Coordination

**CommNet**: Agents share hidden states through communication channels during learning.

**QMIX**: Learn decentralized policies with centralized training:
$Q_{\text{tot}}(\boldsymbol{\tau}, \mathbf{a}) = g_{\psi}(Q_1(\tau_1, a_1), \ldots, Q_N(\tau_N, a_N))$

where $g_{\psi}$ is a mixing network that enforces monotonicity: $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0$. This ensures individual greedy action selection is consistent with team-optimal actions.

## Sample Efficiency and Data Requirements

A major challenge in RL is **sample efficiency** — how much experience is needed to learn?

**Sample complexity**: Number of environment interactions to achieve $\epsilon$-optimal policy.

**Factors affecting sample efficiency**:
- **Discount factor** $\gamma$: Larger $\gamma$ requires more samples
- **State space size**: Exponentially harder for larger spaces
- **Stochasticity**: Noisy environments require more samples
- **Function approximation**: Can improve or harm efficiency depending on representation

**Improving sample efficiency**:
- Model-based methods (leverage learned models)
- Transfer learning (use prior knowledge)
- Auxiliary tasks (learn useful representations)
- Curriculum learning (structured task progression)
- Reward shaping (add intermediate rewards)

## Theoretical Guarantees

### Convergence of Q-Learning

**Theorem** (Watkins & Dayan, 1992): Q-learning converges to $Q^*$ with probability 1 under:
1. All state-action pairs visited infinitely often
2. Learning rates $\alpha_t$ satisfy: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$

**Proof sketch**: Q-learning is a stochastic approximation of the Bellman optimality operator, which is a contraction. The Robbins-Monro conditions on learning rates ensure convergence of the stochastic approximation. □

### Sample Complexity Bounds

For tabular MDPs with $|\mathcal{S}|$ states, $|\mathcal{A}|$ actions:

**PAC bound** (Probably Approximately Correct): With probability $1 - \delta$, algorithms like RMAX or UCRL achieve $\epsilon$-optimal policy after:
$\tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^2(1-\gamma)^3} \log \frac{1}{\delta}\right)$

transitions.

This shows sample complexity is polynomial in problem parameters but inverse polynomial in $\epsilon$ and $(1-\gamma)$.

### Regret Bounds

**Regret** measures suboptimality relative to optimal policy:
$\text{Regret}(T) = \sum_{t=1}^T V^*(s_t) - V^{\pi_t}(s_t)$

Efficient exploration algorithms achieve:
$\text{Regret}(T) = \tilde{O}(\sqrt{|\mathcal{S}||\mathcal{A}| T})$

This is optimal up to logarithmic factors.

## Applications to AI and Mobility Systems

### Autonomous Driving

**State**: Vehicle position, velocity, surrounding vehicles, road geometry
**Actions**: Steering angle, acceleration
**Rewards**: Progress toward destination, comfort, safety

**Approaches**:
- Imitation learning from human drivers (behavioral cloning)
- RL with safety constraints (safe RL)
- Hierarchical RL: high-level routing + low-level control
- Multi-agent RL for coordination in traffic

**Challenges**:
- Safety — cannot learn purely through trial-and-error
- Sim-to-real transfer
- Distribution shift between training and deployment
- Interpretability and verification

### Traffic Signal Control

**State**: Queue lengths, waiting times, vehicle counts
**Actions**: Signal phase selections
**Rewards**: Minimize total delay, maximize throughput

**Classical**: Fixed-time or actuated signals
**RL approaches**:
- Independent Q-learning per intersection
- Multi-agent coordination for network-wide optimization
- Transfer learning across intersections

**Results**: RL-based controllers can reduce delay by 20-40% over fixed-time in simulations.

### Fleet Management and Logistics

**Vehicle repositioning**: Where should idle vehicles wait?
**Dynamic pricing**: How to balance supply and demand?
**Route optimization**: Real-time routing with uncertain demands

**Formulation**: MDP where states are fleet configurations, actions are assignments/prices, rewards are revenue minus costs.

**Challenges**:
- Large state/action spaces (thousands of vehicles)
- Non-stationarity (demand patterns change)
- Coordination among distributed vehicles
- Fairness constraints

**Solutions**:
- Mean field approximations
- Hierarchical RL (regional coordinators + individual vehicles)
- Constrained RL for fairness

### Warehouse Robotics

**Task**: Multi-robot coordination for picking and packing
**State**: Robot positions, inventory levels, pending orders
**Actions**: Movement, picking, placing

**RL advantages**:
- Adapts to changing warehouse layouts
- Learns coordination strategies
- Optimizes multi-objective criteria (throughput, energy, wear)

**Approaches**:
- Centralized training, decentralized execution (CTDE)
- Communication protocols for coordination
- Curriculum learning from simple to complex scenarios

### Energy Management

**Smart charging**: When should EVs charge given electricity prices and grid constraints?
**Building HVAC**: Balance comfort and energy cost

**MDP formulation**:
- State: Battery level, price forecasts, occupancy
- Actions: Charging rates, temperature setpoints
- Rewards: Cost savings, comfort maintenance

**RL advantages**: Learn from data without explicit models, adapt to user preferences

## Connections to Control Theory

RL and optimal control are deeply connected:

| Concept | Control Theory | Reinforcement Learning |
|---------|---------------|------------------------|
| System | Dynamics model $(A, B)$ | Transition $P(s'|s,a)$ |
| Policy | Control law $u = K x$ | Policy $\pi(a|s)$ |
| Value | Cost-to-go $V(x, t)$ | Value function $V^{\pi}(s)$ |
| Optimality | HJB equation | Bellman equation |
| Method | Solve HJB/Riccati | Learn from experience |

**LQR as RL**: The LQR solution is the optimal policy for a special MDP:
- Linear dynamics: $s' = As + Ba + w$
- Quadratic rewards: $r = -(s^T Q s + a^T R a)$
- Optimal policy: $\pi^*(s) = -K s$ where $K = R^{-1} B^T P$

RL generalizes to unknown dynamics, non-linear systems, and general reward functions.

## The Landscape of Modern RL

The field has exploded in recent years:

**Value-based**: DQN, Rainbow, Ape-X
**Policy-based**: REINFORCE, PPO, TRPO
**Actor-critic**: A3C, SAC, TD3
**Model-based**: MBPO, Dreamer, MuZero
**Offline**: CQL, IQL, Decision Transformer
**Multi-agent**: QMIX, MAPPO, MADDPG

**Benchmark environments**:
- Atari games (discrete actions, visual input)
- MuJoCo (continuous control, robotics)
- StarCraft II (multi-agent, partial observability)
- CARLA (autonomous driving simulation)

**State-of-the-art results**:
- Human-level Atari (DQN, Rainbow)
- Complex locomotion (PPO, SAC)
- Strategic games (AlphaGo, OpenAI Five)
- Real robot manipulation (learning from demonstrations + RL)

## Looking Ahead: Model-Based RL and Planning

We've now covered the foundations of sequential decision-making:
- MDPs formalize the interaction between agent and environment
- Value functions characterize optimal behavior via Bellman equations
- RL algorithms learn from experience without knowing the model
- Deep neural networks enable scaling to high-dimensional domains

In our next post, we'll explore **model-based reinforcement learning and planning** in depth. We'll see how:
- Learning environment models enables sample-efficient planning
- Monte Carlo Tree Search combines learned models with look-ahead search
- AlphaZero revolutionized game playing through self-play and search
- Model predictive control bridges classical control and modern RL
- Imagination-based planning enables reasoning about future consequences

Model-based methods represent a synthesis of everything we've learned: they use the dynamic systems perspective to model evolution, the optimization framework to plan actions, the control theory toolkit for trajectory optimization, and the RL paradigm for learning from experience. They also connect to how humans think — we build mental models of the world and use them to imagine and evaluate possible futures before acting.

The journey from MDPs to modern deep RL shows how mathematical foundations enable practical breakthroughs. The Bellman equations, first derived in the 1950s, underpin algorithms that now control robots, optimize logistics, and play complex games. As we move toward increasingly capable AI systems, these mathematical principles continue to guide algorithm design, provide theoretical guarantees, and reveal connections between seemingly different approaches.

---

**References**

**Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

**Bertsekas, D. P.** (2019). *Reinforcement learning and optimal control*. Athena Scientific.

**Puterman, M. L.** (2014). *Markov decision processes: Discrete stochastic dynamic programming*. Wiley.

**Mnih, V., et al.** (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

**Schulman, J., et al.** (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

**Haarnoja, T., et al.** (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *Proceedings of ICML 2018*.

**Silver, D., et al.** (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354–359.

**Agarwal, A., Kakade, S. M., & Yang, L. F.** (2020). Model-based reinforcement learning with a generative model is minimax optimal. *Proceedings of COLT 2020*.