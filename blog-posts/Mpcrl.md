<!-- image: https://andngdtudk.github.io/images/unifreiburg.jpg -->

# Reflection: Fall School on Model Predictive Control and Reinforcement Learning

*University of Freiburg, October 6-10, 2025*

I recently had the incredible opportunity to attend the Fall School on Model Predictive Control and Reinforcement Learning at the University of Freiburg, organized by Prof. Dr. Joschka Boedecker, Prof. Dr. Moritz Diehl, and Prof. Dr. Sebastien Gros from NTNU Trondheim. This intensive five-day program brought together over 100 participants from across Europe and beyond to explore the fascinating intersection of two powerful control paradigms.

## Overview and Motivation

The school addressed a fundamental question in modern control theory: how can we combine the model-driven, constraint-oriented approach of Model Predictive Control (MPC) with the data-driven, performance-oriented methods of Reinforcement Learning (RL)? These two communities have historically operated somewhat independently, but their synthesis offers tremendous potential for solving complex real-world control problems.

## Day 1: Foundations

### Introduction to Reinforcement Learning

The school began with Prof. Boedecker introducing us to the fundamentals of RL through the lens of Markov Decision Processes (MDPs). An MDP is defined as a 4-tuple ⟨S, A, P, r⟩, where S represents the state space, A the action space, P the transition probability function, and r the reward function. The key insight is the Markov property: the future is independent of the past given the present.

We learned about value functions and action-value functions (Q-functions), which quantify how good it is to be in a particular state or to take a particular action. The Bellman equation provides the fundamental relationship:

$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[r(s,a) + \gamma V^{\pi}(s')]$$

The policy iteration and value iteration algorithms were presented as classical methods for solving MDPs exactly in the tabular case. However, the curse of dimensionality quickly becomes prohibitive for continuous or high-dimensional state spaces.

### Dynamic Systems and Simulation

Prof. Diehl introduced us to dynamic system models, explaining how continuous-time ODEs ($\dot{x} = f_c(x,u)$) can be discretized for practical computation. We explored various numerical integration schemes, from simple Euler methods to higher-order Runge-Kutta methods. The RK4 method, with its balance of accuracy and computational cost, emerged as particularly effective for typical control applications.

A crucial distinction was made between state-space models and input-output models. While state-space representations are elegant, real systems often only provide measured outputs $y_k = g(x_k, u_k)$, not complete state information. This led to discussions of system identification and state estimation challenges that would recur throughout the week.

### Basics of Optimization

Andrea Ghezzi provided a comprehensive overview of optimization theory essential for understanding MPC. We covered the hierarchy of optimization problems:

- **Linear Programming (LP)**
- **Quadratic Programming (QP)**
- **Nonlinear Programming (NLP)**
- **Mixed-Integer Nonlinear Programming (MINLP)**

The first-order necessary conditions (FONC) for optimality require $\nabla F(w^*) = 0$, while second-order sufficient conditions (SOSC) require positive definiteness of the Hessian. For constrained problems, the Karush-Kuhn-Tucker (KKT) conditions generalize these concepts, introducing Lagrange multipliers for equality constraints and complementarity conditions for inequalities.

## Day 2: Temporal Difference Methods and Dynamic Programming

### TD Learning and Function Approximation

Prof. Boedecker dove deeper into practical RL algorithms. The TD(0) update rule elegantly combines the simplicity of one-step updates with the power of bootstrapping:

$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

The TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ drives learning, allowing us to update value estimates without waiting for episode completion.

For control, we need to learn action-value functions. Q-learning emerged as a powerful off-policy algorithm:

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_{a'} Q(S',a') - Q(S,A)]$$

The introduction of function approximation was crucial. With Deep Q-Networks (DQN), we can handle continuous state spaces by approximating $Q(s,a)$ with neural networks. Key innovations include:

- **Experience replay** to break correlation in training data
- **Target networks** to stabilize learning
- **Clipping or normalization** for numerical stability

### Dynamic Programming and LQR

Prof. Diehl presented dynamic programming as a principled approach to solving finite-horizon optimal control problems. The cost-to-go function $J_k(x)$ satisfies the recursion:

$$J_k(x) = \min_a [c(x,a) + J_{k+1}(f(x,a))]$$

For linear-quadratic problems, this yields the discrete-time Riccati equation—an analytical solution that forms the basis of Linear Quadratic Regulator (LQR) design. The elegance of the LQR solution, where the optimal control is simply $u = -Kx$ with a time-varying gain matrix $K$, demonstrates the power of exploiting problem structure.

The pandemic control example was particularly illuminating, showing how DP can handle complex decision-making with nonlinear dynamics and state-dependent costs. However, the curse of dimensionality was evident: exact DP requires tabulating $J_k(s)$ for all states, which becomes computationally infeasible as state dimension grows.

## Day 3: Constrained Nonlinear Optimization

Prof. Diehl's lecture on constrained optimization provided the mathematical foundation for MPC. The key insight is that MPC problems, after discretization, become Nonlinear Programs:

$$
\begin{align}
\min_{x,u} &\sum_{k=0}^{N-1} \ell(x_k, u_k) + E(x_N) \\
\text{s.t. } &x_0 = \bar{x}_0 \\
&x_{k+1} = f(x_k, u_k) \\
&h(x_k, u_k) \leq 0
\end{align}
$$

Sequential Quadratic Programming (SQP) emerges as the workhorse algorithm, iteratively solving QP subproblems that linearize the constraints and use a quadratic model of the Lagrangian. Each SQP iteration solves:

$$
\begin{align}
\min_{\Delta w} &\nabla F(w_k)^T \Delta w + \frac{1}{2}\Delta w^T \nabla^2 L(w_k,\lambda_k) \Delta w \\
\text{s.t. } &G(w_k) + \nabla G(w_k)^T \Delta w = 0 \\
&H(w_k) + \nabla H(w_k)^T \Delta w \geq 0
\end{align}
$$

Interior point methods offer an alternative approach, using barrier functions to handle inequalities. The logarithmic barrier $-\tau\sum\log(H_i(w))$ smooths the complementarity conditions, allowing Newton-type methods to be applied.

### Sensitivity Computation

A crucial insight for learning-based MPC: we can compute how the optimal solution changes with parameters using the Implicit Function Theorem. For a parameter-dependent problem with solution $z(p)$, the sensitivity is:

$$\frac{dz(p)}{dp} = -\left[\frac{\partial R}{\partial z}(z,p)\right]^{-1} \frac{\partial R}{\partial p}(z,p)$$

This is essential for gradient-based learning over MPC, as it allows us to backpropagate through the optimization problem. The computation can be done efficiently by solving a linear system rather than re-solving the full NLP for perturbed parameters.

## Day 4: Actor-Critic Methods

Prof. Boedecker introduced policy gradient methods, which directly parameterize and optimize the policy $\pi_\theta(a|s)$. The policy gradient theorem provides a surprising result: we can estimate the gradient of expected return without knowing the system dynamics!

### REINFORCE and Baselines

The REINFORCE algorithm uses the log-derivative trick:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]$$

A baseline $b(s)$ can reduce variance without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) (Q^\pi(s,a) - b(s))]$$

Using $V^\pi(s)$ as the baseline gives us the advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, which has lower variance.

### Modern Actor-Critic Algorithms

Actor-critic methods maintain both a policy (actor) and value function (critic). We explored several state-of-the-art algorithms:

#### Proximal Policy Optimization (PPO)

PPO uses a clipped surrogate objective to limit policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

where $r_t(\theta) = \pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$ is the probability ratio.

#### Deep Deterministic Policy Gradient (DDPG)

DDPG extends DQN to continuous action spaces by learning a deterministic policy $\mu_\theta(s)$ alongside a critic $Q_w(s,a)$. The deterministic policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_a Q_w(s,a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)]$$

#### Soft Actor-Critic (SAC)

SAC adds entropy regularization, maximizing both return and policy entropy:

$$J(\pi) = \mathbb{E}\left[\sum_t R_t + \alpha H(\pi(\cdot|s_t))\right]$$

This encourages exploration and has proven remarkably robust across diverse tasks.

## Day 5: Synthesis of MPC and RL

### Comparing the Two Paradigms

Dirk Reinhardt and Jasper Hoffmann provided a comprehensive taxonomy of how MPC and RL can be combined. The key insight: MPC and RL are complementary, with nearly orthogonal strengths and weaknesses.

**MPC excels at:**
- Handling constraints explicitly
- Incorporating domain knowledge
- Providing interpretable plans
- Adapting to changing objectives

**RL excels at:**
- Learning from experience
- Handling stochastic dynamics implicitly
- Scaling to high dimensions
- Achieving optimal performance

### Inference Architectures

Several architectural patterns emerged:

1. **Parameterized**: Parameters $\theta$ are constant, independent of state
   - Simplest structure
   - No NN evaluation during inference
   - Limited flexibility

2. **Hierarchical**: NN provides input to MPC via $\phi = \phi_\theta(s)$
   - NN evaluated before optimization
   - Doesn't affect optimization structure
   - Used for reference generation, cost modification

3. **Integrated**: Parameters depend on both state and decision variables
   - NN inside optimization problem
   - Most expressive but computationally demanding
   - Used for learned dynamics models

4. **Parallel**: NN and MPC operate independently
   - MPC solution plus NN correction
   - No differentiation through MPC needed
   - Risk of unsafe actions

5. **Algorithmic**: RL guides optimization algorithm
   - Warm-starting, active-set prediction
   - Improves computational efficiency

### Learning Paradigms

Two fundamental approaches to learning in MPC:

#### Aligned Learning

Learn MPC components to match reality:
- **Model learning**: $\min_\theta \mathcal{L}(s' - f_\theta(s,a))$
- **Value function learning**: $V_\theta^{MPC} \approx V^*$
- Maintains interpretability
- May be suboptimal due to model limitations

#### Closed-Loop Learning

Learn to optimize performance directly:
- **Direct optimization**: $\min_\theta J(\pi_\theta^{MPC})$
- Treats MPC as differentiable layer
- Better performance but less interpretable
- Model may not represent true system

### Taxonomy of Combinations

The instructors presented a clear taxonomy:

| Role of MPC | Used in Training | Used in Deployment | Purpose |
|-------------|------------------|-------------------|----------|
| **Expert Actor** | ✓ | ✗ | Generate training data for imitation |
| **Deployed Policy** | ✓ | ✓ | Direct control with learning |
| **Critic** | ✓ | ✗ | Evaluate actions for RL |
| **Safety Filter** | ✗ | ✓ | Post-process RL actions |
| **Reference Generator** | ✗ | ✓ | Provide targets for RL |

Each approach has its place depending on the application requirements and available resources.

## An MPC Prior for SAC

Jasper Hoffmann presented his research on integrating MPC into Soft Actor-Critic. The key insight: parameterize the MPC problem and learn these parameters with RL.

### Design Decisions

Three critical choices:

1. **OCP Formulation**: Determines the realizable action set 
   $$\mathcal{A}_{MPC}(s) = \{u_0^*(s,\phi) \mid \phi \in \Phi\}$$

2. **Critic Type**:
   - **Action Critic** $Q_w^A(s,a)$: Provides feedback on actions, requires differentiable MPC
   - **Parameter Critic** $Q_w^\Phi(s,\phi)$: Provides feedback on parameters, treats MPC as environment

3. **Exploration Strategy**:
   - **Parameter Noise**: $\phi_\theta(s,\xi) = \mu_\theta(s) + \Sigma_\theta^{1/2}\xi$, then $a = u_0^*(s,\phi)$
   - **Action Noise**: $a = u_0^*(s,\phi_\theta(s)) + \xi$

### SAC-ZOP and SAC-FOP

Two algorithms were proposed:

#### SAC-ZOP (Zero-Order, Parameter Noise)
- Uses parameter critic $Q^\Phi_w(s,\phi)$
- No differentiation through MPC during training
- Faster training, simpler implementation

#### SAC-FOP (First-Order, Parameter Noise)
- Uses action critic $Q^A_w(s,a)$
- Differentiates through MPC for actor updates
- Leverages sensitivity information

**Results**: Both significantly outperformed vanilla SAC and TD-MPC2 baselines across multiple environments, with SAC-ZOP being 3× faster to train than SAC-FOP while achieving comparable performance. Crucially, parameter noise exploration maintained 0% constraint violations compared to 65% for action noise.

## Imitation Learning from MPC

Andrea Ghezzi explored how to learn policies that imitate MPC controllers. This is essentially explicit MPC—representing the implicit policy $\pi_{MPC}(s) = u_0^*(s)$ explicitly.

### Loss Functions

Several loss functions were discussed:

#### Behavioral Cloning (L2 loss)
$$\mathcal{L}_2(\theta) = \mathbb{E}_{s\sim\mathcal{D}}[(\pi(s;\theta) - \pi^*(s))^2]$$

#### Exact Q-Loss
Fix first control and evaluate:
$$Q_{MPC}(s,a) = \min_{x,u} \sum\ell(x_k,u_k) + T(x_N)$$
subject to $x_0=s$, $u_0=a$, dynamics, constraints

The gradient $\nabla_u Q(s,u)|_{u=\pi(s;\theta)}$ comes from the Lagrange multiplier of the $u_0$ constraint.

#### Sobolev Training
Include sensitivities in training data:
$$\mathcal{L}_{sob} = \mathbb{E}[(u-\pi(x;\theta))^2 + \alpha(\partial u/\partial x - \partial\pi/\partial x|_x)^2]$$

#### PlanNetX
Match full MPC trajectories:
$$\mathcal{L}_p = \mathbb{E}\left[\frac{1}{N}\sum_k \gamma^k \|\hat{x}_k(\pi;x_0,\theta) - x_k^*\|^2\right]$$

### Data Collection and Safety

The **covariate shift problem** is critical: if the policy never learns to recover from its own mistakes, it will fail during deployment. **DAgger** (Dataset Aggregation) addresses this by rolling out the learned policy, collecting expert labels for visited states, and retraining.

For safety during learning, several approaches were discussed:
- MPC safety filters to project learned actions onto safe set
- Activation pattern analysis for ReLU networks to ensure local stability
- Constraint handling through penalty methods

## Why Does RL Over MPC Work?

Prof. Gros provided the theoretical foundation in what was perhaps the most conceptually deep lecture of the week.

### MPC as a Model of MDPs

The key insight: **MPC can be viewed as modeling the optimal Q-function of the real-world MDP, not just approximating the system dynamics!**

For a world MDP with optimal $Q^*$, we can construct a model MDP with $Q^{MPC}$ such that $Q^{MPC}(s,a) = Q^*(s,a)$ even if the prediction model $f_\theta$ doesn't accurately match the real dynamics.

**Theorem**: Under technical conditions, for a richly parameterized MPC:

$$
\begin{align}
\min_{x,u} &T_\theta(x_N) + \sum \gamma^k L_\theta(x_k,u_k) \\
\text{s.t. } &x_0=s, \, x_{k+1}=f_\theta(x_k,u_k), \, h_\theta(x_k,u_k)\leq 0
\end{align}
$$

there exists $\theta$ such that $Q^{MPC}_\theta(s,a) = Q^*(s,a)$ for all $s,a$.

This is profound: **we can compensate for model deficiencies by adjusting the cost and constraints!** The "best model for control" is not necessarily the "best model to fit data."

### When is RL Most Beneficial?

Based on the theory, RL over MPC provides the most benefit when:

1. **Economic problems** with low dissipativity (trajectories spread over state space)
2. **Active constraints**: optimal steady-state near/at constraint boundaries
3. **Varying exogenous inputs**: changing prices, references, disturbances
4. **Task-based problems**: racing, minimum-time, complex objectives
5. **Model insufficiency**: prediction model cannot capture relevant state transitions
6. **Non-smooth problems**: value function curvature changes significantly

Conversely, RL provides less benefit for:
- Smooth tracking problems near fixed steady states
- Systems spending most time away from constraints
- Problems where LQR-type solutions are near-optimal

### The State Estimation Challenge

A critical assumption in the theory: the world MDP and model MDP must share the same state representation. In practice, we often don't have access to the true Markov state—only a history of observations.

Solutions include:

1. **Input-output models**: ARX, multi-step predictors
2. **Latent states**: AI-driven compressed representations
3. **State observers**: Model-based estimation

The key is that state estimation should be part of the learning loop:
$$\pi^\mathcal{D}_\theta(\text{Data}) = \pi_\theta(\phi_\theta(\text{Data}))$$

The gradient becomes:
$$\nabla_\theta \pi^\mathcal{D}_\theta(\text{Data}) = \nabla_\theta \pi_\theta(s) + \nabla_\theta \phi_\theta(\text{Data}) \nabla_s \pi_\theta(s)$$

This allows us to optimize both the controller and observer jointly.

## Software Tools: leap-c

Throughout the week, we used **leap-c** (Learning for Predictive Control), an open-source Python library that integrates acados with PyTorch. Key features include:

- Seamless MPC integration as differentiable PyTorch layer
- Efficient batched computation with multithreading
- Sensitivity computation for parameters, states, and Q-functions
- Support for both aligned and closed-loop learning
- Modular architecture for easy experimentation

The library is actively developed and available at [github.com/leap-c/leap-c](https://github.com/leap-c/leap-c).

## Practical Insights and Examples

### Home Energy Management

A simulation study demonstrated the power of RL over MPC for energy systems. Starting with classical system identification to build an MPC model, then applying RL tuning resulted in significant performance improvements. The RL phase optimized parameters in the cost function, constraints, and model to better handle stochasticity and varying electricity prices.

### Challenges and Limitations

The instructors were admirably transparent about challenges:

1. **Computational cost**: Optimization is more expensive than neural network inference
2. **GPU deployment**: MPC solvers lag behind in GPU utilization
3. **Non-convexity**: Local minima can impede learning
4. **High-dimensional parameter spaces**: Especially for multi-step predictive control
5. **Sample efficiency**: While better than pure RL, still requires substantial data

## Personal Reflections

### Theoretical Depth

The theoretical framework connecting MDPs, dynamic programming, and MPC was revelatory. Understanding MPC as a model of $Q^*$ rather than just a model-based controller fundamentally changes how we think about learning and control. The sufficient conditions for MPC optimality, involving the value function and conditional distribution, explain why classical system identification may not yield optimal policies.

### Practical Promise

The empirical results were compelling. SAC-ZOP and SAC-FOP demonstrated that MPC priors can dramatically improve sample efficiency and safety in RL. The fact that parameter critics worked so well without differentiating through MPC was surprising and practically important.

### Interdisciplinary Challenges

The school highlighted the cultural differences between MPC and RL communities—different notations ($x$ vs $s$, $u$ vs $a$), different assumptions (deterministic vs stochastic), different priorities (constraints vs performance). Bridging these gaps requires effort but yields powerful hybrid methods.

### Open Questions

Several exciting research directions emerged:

1. **Belief-state MPC**: How to handle partial observability optimally?
2. **Multi-agent systems**: Extending the framework to coordination and communication
3. **Risk-aware learning**: Moving beyond expectation to consider mission success probabilities
4. **Computational efficiency**: Scaling to high-dimensional problems with massive datasets
5. **Theoretical guarantees**: Refining conditions for convergence and optimality

## Conclusion

This fall school provided both breadth and depth, covering everything from fundamental RL algorithms to cutting-edge research on learned MPC. The synthesis of MPC and RL represents more than a technical combination—it's a philosophical bridge between model-driven and data-driven paradigms.

### Key Takeaways

1. **MPC and RL are complementary**: MPC provides structure, constraints, and interpretability; RL provides performance optimization and adaptation.

2. **Learning for control ≠ Learning for prediction**: The best model for control performance may not be the best model to fit data.

3. **Hierarchical architectures work**: Parameterizing MPC and learning with RL provides a practical path forward.

4. **Theory guides practice**: Understanding MPC as modeling $Q^*$ provides principled ways to design learnable controllers.

5. **Tools are maturing**: Software like leap-c makes research and application increasingly accessible.

As autonomous systems become more complex and operate in less structured environments, the ability to combine domain knowledge (MPC) with learning from experience (RL) will be essential. This school provided not just technical skills but a conceptual framework for thinking about this integration.

I'm grateful to the organizers and instructors for creating such a rich learning experience, and to my fellow participants for thought-provoking discussions. The field of learning-based MPC is rapidly evolving, and I'm excited to contribute to its development.

---

## Resources

- **leap-c repository**: [github.com/leap-c/leap-c](https://github.com/leap-c/leap-c)
- **acados**: [docs.acados.org](https://docs.acados.org)
- **Fall School Materials**: Available from organizers upon request

## Acknowledgments

Special thanks to:
- Prof. Dr. Joschka Boedecker (University of Freiburg)
- Prof. Dr. Moritz Diehl (University of Freiburg)
- Prof. Dr. Sebastien Gros (NTNU Trondheim)
- Teaching assistants: Leonard Fichtner, Andrea Ghezzi, and Jasper Hoffmann


*For those interested in diving deeper, I highly recommend checking out the lecture materials and the leap-c repository. The combination of rigorous theory and practical implementation makes this an excellent entry point into learning-based optimal control.*
