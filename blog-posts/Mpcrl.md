<!-- image: https://andngdtudk.github.io/images/unifreiburg.jpg -->

# Reflections on the Fall School: Model Predictive Control and Reinforcement Learning

*University of Freiburg, October 6–10, 2025*

I recently had the privilege of attending a comprehensive fall school on Model Predictive Control (MPC) and Reinforcement Learning (RL) at the University of Freiburg, taught by **Joschka Boedecker**, **Moritz Diehl**, and **Sebastien Gros**. This intensive week-long program brought together two communities that have traditionally operated independently—control theory and machine learning—and explored how their synthesis can lead to more powerful approaches for sequential decision-making problems. In this reflection, I'll share my key takeaways from this transformative experience.

<p align="center">
    <img src="images/Mpcrl25-large_bw.jpg" alt="Group photo at the Fall School on Model Predictive Control and Reinforcement Learning, University of Freiburg, October 2025" style="display: block; margin: auto; width: 65%;">
</p>

<figcaption style="text-align: center;">
    Figure 1. Group photo at the Fall School on Model Predictive Control and Reinforcement Learning, University of Freiburg, October 2025
    <br>
    <span style="font-size: 0.9em;">
        An inspiring week of lectures, discussions, and collaboration with researchers and students from around the world, exploring how Model Predictive Control and Reinforcement Learning can jointly advance intelligent decision-making.
    </span>
    <br>
    <span style="font-size: 0.8em; font-style: italic;">
        Photo courtesy of the University of Freiburg
    </span>
</figcaption>


---

## The Fundamental Framework: MDPs as a Unifying Language

The course began by establishing **Markov Decision Processes (MDPs)** as the theoretical foundation connecting both MPC and RL.  
An MDP is defined by a four-tuple ⟨S, A, P, r⟩, consisting of states, actions, transition probabilities, and rewards.  

What struck me most was how this seemingly simple framework provides such a general way to describe sequential decision-making problems, accommodating both the stochastic nature of RL and the deterministic planning perspective of MPC.

The **Markov property**—that the future is independent of the past given the present—is crucial here. It allows us to make decisions based solely on the current state without needing to track the entire history of the system. The goal in an MDP is to find a policy π that maximizes the expected discounted cumulative reward:

\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
\]

where the discount factor \( \gamma \in (0, 1) \) ensures mathematical tractability and reflects our preference for immediate rewards over distant ones.

---

## Dynamic Programming: The Theoretical Backbone

One of the most elegant concepts we explored was **dynamic programming (DP)** and its connection to optimal control.  
The **Bellman equation** provides a recursive relationship between the value of a state and the values of its successor states:

\[
V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [r(s,a) + \gamma V_\pi(s')]
\]

This embodies the *principle of optimality*: an optimal policy must remain optimal after any initial decision.

We learned two fundamental algorithms for solving MDPs:
- **Policy iteration** alternates between evaluating a policy and improving it.
- **Value iteration** combines both steps using the Bellman optimality equation:

\[
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[r(s,a) + \gamma V_k(s')]
\]

Despite their simplicity, these algorithms suffer from the **curse of dimensionality**, which motivates the need for approximation methods in RL and online optimization in MPC.

---

## Model Predictive Control: Planning in Action

MPC tackles the optimal control problem by solving an optimization problem **online** at every time step:

\[
\min_{x,u} \; T(x_N) + \sum_{k=0}^{N-1} L(x_k, u_k)
\]
subject to:
\[
x_{k+1} = f(x_k, u_k), \quad h(x_k, u_k) \le 0, \quad x_0 = s
\]

Only the first control \( u_0^* \) is applied, and the process repeats—this *receding horizon* strategy is central to MPC.

The connection between DP and MPC became clear through the cost-to-go recursion:

\[
J_k(x) = \min_a \, c(x,a) + J_{k+1}(f(x,a))
\]

and its Q-function form:

\[
Q_k(s,a) = c(s,a) + J_{k+1}(f(s,a))
\]

For **linear-quadratic systems**, this leads to the **Riccati recursion** and the celebrated **LQR** controller \( u = -Kx \), which offers elegant closed-form feedback laws.

---

## Model-Free Reinforcement Learning: Learning from Experience

RL learns optimal behavior directly from interaction.  
The **temporal difference (TD)** method bridges Monte Carlo estimation and DP through the update:

\[
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
\]

The TD error \( \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \) quantifies the mismatch between predicted and actual returns.  
For control, **Q-learning** directly estimates the optimal action-value function:

\[
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
\]

This separation between the **target policy** and **behavior policy** (ε-greedy exploration) balances exploration and exploitation effectively.

---

## Scaling with Function Approximation

To handle large state spaces, we replace tables with parameterized approximators.  
The **semi-gradient TD(0)** update is:

\[
w \leftarrow w + \alpha [R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)] \nabla \hat{v}(S_t, w)
\]

Deep Q-Networks (DQN) extend this with neural networks, **experience replay**, and **target networks**, stabilizing learning in complex domains.

---

## Actor-Critic Methods and Policy Gradients

The **policy gradient theorem** provides the foundation for actor-critic algorithms:

\[
\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(A|S) Q_\pi(S,A)]
\]

Actor-critic methods combine:
- A **critic** estimating \( Q_\pi(s,a) \)
- An **actor** updating the policy parameters \( \theta \)

We explored modern methods:
- **PPO** for stable updates  
- **DDPG** for deterministic continuous control  
- **SAC** for entropy-regularized learning

---

## The Synthesis: Bringing MPC and RL Together

The synthesis lectures (Dirk Reinhardt & Jasper Hoffmann) emphasized MPC and RL as **complementary paradigms**:
- MPC brings **models, safety, and constraints**.
- RL brings **adaptivity and data-driven learning**.

A taxonomy of hybrid approaches was discussed:
1. **MPC as an expert** (for imitation or reward shaping)  
2. **MPC inside the policy** (parameterized or hierarchical integration)  
3. **MPC and RL co-training** (jointly optimizing model, cost, and policy)

The key philosophical divide between **aligned learning** (model fidelity) and **closed-loop learning** (empirical performance) highlighted when learning should adapt beyond physical interpretability.

---

## Practical Implementation: SAC-ZOP and SAC-FOP

In **SAC-ZOP** and **SAC-FOP**, MPC acts as a **prior inside Soft Actor-Critic**.  
The policy network outputs MPC parameters φ, and the MPC then solves:

\[
\min_{x,u} T_\theta(x_N) + \sum_{k=0}^{N-1} L_\theta(x_k, u_k)
\]

The first control \( u_0^* \) becomes the executed action.

- **SAC-ZOP** avoids differentiating through MPC (zero-order).  
- **SAC-FOP** differentiates through the MPC solution (first-order).

Both methods maintain **safety during exploration** and achieve superior **sample efficiency**, with SAC-ZOP performing competitively despite not computing gradients through MPC.

---

## Imitation Learning from MPC

**Andrea Ghezzi** introduced imitation learning as a way to replace online MPC with learned policies.  
Beyond behavioral cloning, advanced techniques include:

- **Exact Q-loss** — using MPC’s cost structure to train via Bellman consistency.  
- **Sobolev training** — learning both actions and derivatives ∂u/∂x.  
- **Data augmentation** — generating synthetic nearby samples via NLP sensitivities.  
- **DAgger** — iteratively aggregating data from rollouts for covariate shift correction.  

Safety mechanisms such as **safety filters** or **stability-constrained networks** ensure reliable deployment of learned policies.

---

## Theoretical Foundations: Why Does It Work?

Prof. Gros’s lecture provided deep theoretical justification for **learning over MPC**.  
Even with imperfect models, MPC can approximate the optimal Q-function if its cost and constraints are parameterized correctly.

This means **costs can compensate for model deficiencies**—the MPC doesn’t need to predict the future perfectly, only to **choose good actions**.  

The conditions for MPC optimality highlight when **RL provides true value**—particularly in:
- Economic MPC or variable objectives  
- Systems operating near constraints  
- Low-dissipativity dynamics (wide state distributions)

---

## The State Estimation Challenge

A key assumption in MPC is full state observability.  
In reality, controllers often operate on estimated or latent states.

Approaches include:
- **Input-output models** (e.g., ARX, multi-step predictors)  
- **Latent embeddings** (learned compact representations)  
- **Observers** like Kalman filters or Moving Horizon Estimation (MHE)

When the state estimator and policy are both learned, their gradients interact:

\[
\nabla_\theta \pi^D_\theta(\text{Data}) = \nabla_\theta \pi_\theta(s) + \nabla_\theta \phi_\theta(\text{Data}) \nabla_s \pi_\theta(s)
\]

This integrates perception and control into a single optimization loop.

---

## Optimization and Numerical Methods

We explored **numerical methods** for discretization and nonlinear programming:
- Runge–Kutta integrators for accurate discretization  
- **Sequential Quadratic Programming (SQP)**  
- **Interior Point Methods**  
- **Sensitivity analysis** via the implicit function theorem  

The **leap-c** framework implements differentiable MPC layers compatible with **PyTorch**, bridging control and deep learning toolchains.

---

## Challenges and Open Questions

Despite rapid progress, several challenges remain:
- Computational scalability to high-dimensional RL problems  
- Risk-aware and distributionally robust MPC-RL methods  
- Multi-agent coordination and decentralized control  
- Safety verification of neural components  
- Generalization to unseen conditions  

Balancing **safety, optimality, and generalization** remains a delicate problem, and hybrid MPC-RL systems provide a promising middle ground.

---

## Reflections and Future Directions

This school reshaped how I think about learning and control.  
MPC and RL are not competing but **complementary philosophies**—one grounded in models, the other in data.

The paradigm shift from **model fitting** to **policy learning** reframes control as optimizing for *action quality* rather than *prediction accuracy*.  

Future directions include:
- **Belief-state MPC** for uncertainty-aware planning  
- **World models optimized for control**  
- **Compositional MPC-RL architectures**  
- **Transfer and meta-learning** for adaptive controllers  

---

## Conclusion

The Fall School provided a deep and coherent understanding of the synergy between MPC and RL.  
It highlighted a unifying principle: **intelligent systems must both learn and reason**.

By merging MPC’s structured optimization with RL’s adaptive learning, we can design controllers that are safe, efficient, and generalizable.

I am deeply grateful to the instructors and participants for this intellectually rich experience.  
The journey from **Markov Decision Processes to modern MPC-RL synthesis** has been both challenging and inspiring, and it has strengthened my vision for advancing intelligent transportation and control systems.

---

