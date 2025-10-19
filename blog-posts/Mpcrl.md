<!-- image: https://andngdtudk.github.io/images/unifreiburg.jpg -->
# Reflections on the Fall School: Model Predictive Control and Reinforcement Learning

*University of Freiburg, October 6-10, 2025*

I recently had the privilege of attending a comprehensive fall school on Model Predictive Control (MPC) and Reinforcement Learning (RL) at the University of Freiburg, taught by Joschka Boedecker, Moritz Diehl, and Sebastien Gros. This intensive week-long program brought together two communities that have traditionally operated independently—control theory and machine learning—and explored how their synthesis can lead to more powerful approaches for sequential decision-making problems. In this reflection, I'll share my key takeaways from this transformative experience.

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

## The Fundamental Framework: MDPs as a Unifying Language

The course began by establishing Markov Decision Processes (MDPs) as the theoretical foundation connecting both MPC and RL. An MDP is defined by a four-tuple $\langle S, A, P, r \rangle$, consisting of states, actions, transition probabilities, and rewards. What struck me most was how this seemingly simple framework provides such a general way to describe sequential decision-making problems, accommodating both the stochastic nature of RL and the deterministic planning perspective of MPC.

The Markov property—that the future is independent of the past given the present—is crucial here. It allows us to make decisions based solely on the current state without needing to track the entire history of the system. The goal in an MDP is to find a policy $\pi$ that maximizes the expected discounted cumulative reward, which we express as the return:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

The discount factor $\gamma \in (0,1)$ ensures that infinite horizon problems remain mathematically tractable and reflects our preference for immediate rewards over distant ones.

## Dynamic Programming: The Theoretical Backbone

One of the most elegant concepts we explored was dynamic programming and its connection to optimal control. The Bellman equation provides a recursive relationship between the value of a state and the values of its successor states. For a policy $\pi$, the state-value function satisfies:

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[r(s,a) + \gamma V_\pi(s')]$$

This equation embodies the principle of optimality: an optimal policy has the property that whatever the initial state and decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

We learned two fundamental algorithms for solving MDPs when the dynamics are known. Policy iteration alternates between evaluating a policy (computing $V_\pi$) and improving it by acting greedily with respect to the current value function. Value iteration combines these steps, directly updating the value function using the Bellman optimality equation:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[r(s,a) + \gamma V_k(s')]$$

What fascinated me was how these algorithms, despite their conceptual simplicity, suffer from the "curse of dimensionality"—a phrase coined by Richard Bellman himself. Exact dynamic programming requires tabulating the value function for all possible states, which becomes computationally infeasible as state dimensionality grows. This limitation motivates both the approximation methods in RL and the online optimization approach in MPC.

## Model Predictive Control: Planning in Action

MPC approaches the optimal control problem from a different angle. Rather than computing a policy offline for all possible states, MPC repeatedly solves an optimal control problem online at each time step. At the current state $s$, MPC solves:

$$\begin{aligned}
\min_{x,u} \quad & T(x_N) + \sum_{k=0}^{N-1} L(x_k, u_k) \\
\text{subject to:} \quad & x_{k+1} = f(x_k, u_k) \\
& h(x_k, u_k) \leq 0 \\
& x_0 = s
\end{aligned}$$

The controller applies only the first action $u_0^*$ and repeats this process at the next state. This "receding horizon" strategy is remarkably effective in practice, despite throwing away most of the computed plan.

The lectures on dynamic programming for finite horizons deepened my understanding of MPC's theoretical foundations. The cost-to-go function $J_k(x)$ represents the minimum cost achievable from state $x$ at time $k$. Through backward recursion starting from the terminal cost, we can compute:

$$J_k(x) = \min_a c(x,a) + J_{k+1}(f(x,a))$$

The Q-function provides a natural way to extract the optimal feedback control policy:

$$Q_k(s,a) = c(s,a) + J_{k+1}(f(s,a))$$
$$\pi^*_k(s) = \arg\min_a Q_k(s,a)$$

For linear-quadratic problems, this recursion takes a special form. When the dynamics are linear ($x_{k+1} = Ax_k + Bu_k$) and the cost is quadratic, the value function remains quadratic throughout the backward pass, leading to the famous Riccati recursion. This yields the Linear Quadratic Regulator (LQR), where the optimal feedback is simply a linear state feedback: $u = -Kx$. The algebraic Riccati equation provides the infinite-horizon solution, requiring only a single matrix-vector multiplication at each time step. This computational efficiency makes LQR attractive for real-time control, though its linearity assumption limits applicability to systems operating near equilibrium points.

## Model-Free Reinforcement Learning: Learning from Experience

While MPC assumes knowledge of system dynamics, RL takes a fundamentally different approach: learning optimal behavior directly from interaction with the environment. The course introduced temporal difference (TD) learning as a bridge between Monte Carlo methods and dynamic programming.

The simplest TD algorithm, TD(0), updates the value function after each step using the observed reward and the current value estimate of the next state:

$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

The term $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error, representing the difference between our prediction and a better estimate based on the actual transition. This bootstrapping—updating estimates based on other estimates—is a key distinction from Monte Carlo methods that must wait until the end of an episode.

For control, we need to learn about actions, not just states. This led us to Q-learning, an off-policy TD algorithm that directly learns the optimal action-value function:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

The elegance of Q-learning lies in its ability to learn the optimal policy while following an exploratory behavior policy, typically $\varepsilon$-greedy. This separation between the target policy (what we're learning) and the behavior policy (what we're doing) is crucial for balancing exploration and exploitation.

## Scaling with Function Approximation

Tabular methods quickly become impractical for large or continuous state spaces. The solution is function approximation—representing value functions or policies with parameterized functions like neural networks. The semi-gradient TD(0) algorithm adapts the tabular version:

$$w \leftarrow w + \alpha[R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)]\nabla \hat{v}(S_t, w)$$

These methods are called "semi-gradient" because they take the gradient with respect to the prediction but not the target, treating the target as fixed. This introduces bias but often works well in practice.

Deep Q-Networks (DQN) demonstrated that deep neural networks could successfully represent action-value functions for high-dimensional problems. The key innovations that stabilized training were experience replay (breaking temporal correlations by randomly sampling from a replay buffer) and target networks (fixing the parameters used to compute targets to prevent oscillations). DQN's success on Atari games from raw pixels marked a watershed moment, showing that deep RL could tackle problems previously considered intractable.

## Actor-Critic Methods and Policy Gradients

The course then moved to actor-critic methods, which maintain explicit representations of both the policy (actor) and value function (critic). The policy gradient theorem provides the foundation:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(A|S)Q^{\pi_\theta}(S,A)]$$

This remarkable result shows we can compute the gradient of expected return by weighting the score function (gradient of log probability) by the action value. The REINFORCE algorithm uses the full return $G_t$ as an unbiased estimate of $Q$, but suffers from high variance. Actor-critic methods reduce variance by using a learned critic to estimate $Q$, though this introduces bias.

We explored several modern algorithms building on these ideas. Proximal Policy Optimization (PPO) constrains policy updates to prevent destructively large steps, using either an adaptive penalty on KL divergence or a clipped surrogate objective. Deep Deterministic Policy Gradient (DDPG) extends Q-learning to continuous action spaces by learning a deterministic policy $\mu_\theta(s)$ alongside a critic $Q_w(s,a)$, updated using the deterministic policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \mu_\theta(S)\nabla_a Q_w(S,a)|_{a=\mu_\theta(S)}]$$

Soft Actor-Critic (SAC) adds entropy regularization to the objective, encouraging exploration and improving robustness. The maximum entropy objective:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t R_t + \alpha \sum_t H(\pi(\cdot|S_t))\right]$$

leads to particularly stable learning dynamics. SAC has become one of the most reliable algorithms for continuous control, balancing sample efficiency with robust performance.

## The Synthesis: Bringing MPC and RL Together

The most thought-provoking part of the school was exploring how to combine MPC and RL. The synthesis lecture by Dirk Reinhardt and Jasper Hoffmann presented a taxonomy of approaches based on when and how MPC components are used.

The fundamental insight is recognizing MPC and RL as complementary rather than competing approaches. MPC brings model-driven planning with explicit constraint handling, while RL offers data-driven learning focused on closed-loop performance. Both ultimately solve sequential decision-making problems but from different philosophical starting points.

The taxonomy organized combination approaches along two axes: what's used during deployment versus what's used during learning. We can use MPC as an expert actor to generate training data, incorporating its safe, model-informed behavior into the learning process. We can integrate MPC within the deployed policy through various architectural patterns—parameterized, parallel, hierarchical, integrated, or algorithmic. Each offers different tradeoffs between computational complexity, safety guarantees, and flexibility.

The distinction between **aligned learning** and **closed-loop learning** proved particularly important. Aligned learning fits individual MPC components (model, cost, constraints) to better approximate the true environment, maintaining interpretability and physical meaning. Closed-loop learning directly optimizes for what works best in practice, allowing the MPC's internal model and costs to become misaligned with reality if that improves performance. This paradigm shift—from requiring model accuracy to requiring policy optimality—challenges conventional thinking in both control and machine learning.

## Practical Implementation: The SAC-ZOP and SAC-FOP Algorithms

Jasper Hoffmann's lecture on integrating an MPC prior into SAC provided concrete algorithmic details. The hierarchical architecture uses a neural network to produce parameters $\phi$ for the MPC, rather than directly outputting actions. The MPC then solves:

$$\begin{aligned}
\min_{x,u} \quad & T_\theta(x_N) + \sum_{k=0}^{N-1} L_\theta(x_k, u_k) \\
\text{subject to:} \quad & x_{k+1} = f_\theta(x_k, u_k) \\
& h_\theta(x_k, u_k) \leq 0 \\
& x_0 = s
\end{aligned}$$

with the first control $u_0^*$ becoming the action applied to the system.

Two variants were presented. **SAC-ZOP** (Zero-Order, Parameter noise) uses a parameter critic $Q^\Phi_w(s,\phi)$ and treats the MPC as part of the environment, avoiding differentiation through the optimization. **SAC-FOP** (First-Order, Parameter noise) uses an action critic $Q^A_w(s,a)$ and differentiates through the MPC solution map to compute policy gradients. Both inject noise in the parameter space before MPC, allowing safe exploration since the MPC filters any unsafe parameter perturbations.

The concept of a parameter MDP proved enlightening. Rather than learning a policy over actions directly, we learn a policy over MPC parameters. This induces an action policy through the MPC solution map. The parameter critic provides feedback on parameter choices, which can be computationally more efficient than action critics since it bypasses MPC during training updates. However, the action critic potentially provides more informed updates by incorporating sensitivities of the MPC scheme.

The experimental results showed both algorithms significantly outperforming vanilla SAC and model-based baselines like TD-MPC2 in sample efficiency, while maintaining perfect safety (zero constraint violations during training). Perhaps surprisingly, SAC-ZOP proved competitive with SAC-FOP despite not differentiating through MPC, suggesting the parameter space provides a useful abstraction for learning.

## Imitation Learning from MPC

Andrea Ghezzi's lecture on imitation learning provided another perspective on learning from MPC. The goal is to approximate the MPC policy with a neural network that can be evaluated much faster than solving the optimization online. Beyond simple behavioral cloning using supervised learning, several sophisticated approaches emerged.

The **exact Q-loss** defines a loss function directly from the MPC optimal control problem. By fixing the first control to the neural network's output and solving the resulting problem, we obtain a cost $Q(s,a)$ that assigns a value to each proposed action. This allows the network to be trained using the Bellman structure rather than just imitating actions. The gradient is given by the Lagrange multiplier corresponding to the control constraint, connecting nicely to concepts from optimization theory.

**Sobolev training** enriches the training data with sensitivity information $\frac{\partial u}{\partial x}$ computed from the MPC, allowing the network to learn not just the policy but also its Jacobian. This can significantly improve performance near training points. Data augmentation using NLP sensitivities addresses the expense of generating MPC solutions by leveraging the implicit function theorem to create synthetic training examples around each solved problem.

The challenge of distribution mismatch—where the network encounters states during deployment that differ from training states—can be addressed through **DAgger** (Dataset Aggregation). This iterative approach collects data by rolling out the learned policy, queries the MPC for correct actions at the encountered states, and retrains on the aggregated dataset. Over iterations, the learned policy sees its own state distribution, closing the covariate shift gap.

Verification and safety remain critical concerns. For linear systems, we can sometimes guarantee that neural network controllers maintain stability by constraining the weights of the final layer to ensure the closed-loop system remains stable near equilibrium. For nonlinear systems, safety filters provide a practical solution—the MPC projects the neural network's proposed action onto the feasible set defined by constraints and dynamics predictions, ensuring safety at the cost of potentially suboptimal actions.

## The Theoretical Foundations: Why Does It Work?

Sebastien Gros's theoretical lecture provided the deepest insights into why learning over MPC is justified and when it's most beneficial. The key theoretical result addresses a fundamental question: can a simplified MPC formulation with limited model fidelity still produce optimal policies?

The answer, surprisingly, is yes—under certain conditions. The MPC can be understood as defining an approximation to the optimal Q-function of the true MDP. For a fully parameterized MPC that allows adjusting the cost function $L_\theta$, constraints $h_\theta$, and model $f_\theta$, there exists a parameterization $\theta$ such that:

$$Q^{MPC}_\theta(s,a) = Q^*(s,a)$$

even if the model cannot accurately describe the real system dynamics.

This result is profound: **we can compensate for model deficiencies through the cost function and constraints**. The MPC becomes a model of the optimal policy, not merely a policy approximation using open-loop predictions. The stage cost and terminal cost encode not just our objectives but also corrections for model errors and the value of future states beyond the horizon.

The sufficient conditions for MPC optimality reveal when classical system identification might fall short. For a model $f$ to yield optimal MPC performance, it must satisfy conditions relating the model to the optimal value function and conditional distribution of state transitions. Standard regression approaches like ridge regression (minimizing prediction error) or maximum likelihood (finding the most probable model) generally don't satisfy these conditions, except in special cases like LQR with process noise.

This explains when RL is most beneficial for MPC. Problems with smooth tracking to fixed setpoints far from constraints, where the system spends most time near equilibrium, see limited gains from learning. But economic MPC with varying objectives, task-based problems with termination conditions, systems operating near constraints, or problems with low dissipativity (trajectories spreading across the state space) can benefit substantially. The more the optimal value function varies and the less it resembles a quadratic form, the more classical model fitting diverges from optimal policy learning.

## The State Estimation Challenge

One critical assumption in the theory bears emphasis: the MPC and the real world must share the same state representation. In practice, we rarely observe the true Markov state directly. This challenge can be addressed through several approaches.

**Input-output models** like ARX formulations or multi-step predictors build the state from recent measurement and action history, avoiding explicit state estimation. The state becomes:

$$s_k = (y_{k-1}, a_{k-1}, \ldots, y_{k-n}, a_{k-n})$$

effectively treating the history as the Markov state.

**Latent state models** use learned embeddings or compressed representations, with neural networks discovering low-dimensional state abstractions from high-dimensional observations. Model-based state observers like moving horizon estimation (MHE) or Kalman filters provide physically meaningful state estimates. The key is bringing state estimation into the learning loop, optimizing the parameters $\theta$ of both the estimator $\phi_\theta$ and the policy $\pi_\theta$ jointly. The policy gradient becomes:

$$\nabla_\theta \pi^D_\theta(\text{Data}) = \nabla_\theta \pi_\theta(s) + \nabla_\theta \phi_\theta(\text{Data})\nabla_s \pi_\theta(s)$$

This unified view treats state estimation as part of the decision-making pipeline rather than a separate preprocessing step.

## Optimization and Numerical Methods

Understanding the optimization machinery underlying MPC proved essential. The course covered how continuous-time optimal control problems are discretized using numerical integration methods. The choice matters—higher-order Runge-Kutta methods like RK4 achieve much better accuracy than simple Euler integration for the same computational cost. These methods transform continuous dynamics into discrete-time models suitable for optimization.

The resulting nonlinear programs (NLPs) are solved using Sequential Quadratic Programming (SQP) or interior point methods. SQP linearizes the KKT optimality conditions at each iteration, solving a sequence of quadratic programs that progressively approach the solution. Interior point methods smooth the complementarity conditions using logarithmic barriers, converting inequalities into smooth equality constraints at the cost of introducing a barrier parameter that must be driven to zero.

The sensitivity analysis—computing how solutions change with parameters—enables differentiation through MPC for gradient-based learning. The implicit function theorem provides:

$$\frac{\partial z^*}{\partial p} = -M^{-1}r$$

where $M$ is the KKT matrix and $r$ contains parameter derivatives. For linear-quadratic problems, this reduces to solving a Riccati equation. For nonlinear problems, we can use either exact sensitivities (solving the sensitivity equations) or approximate them through finite differences or automatic differentiation.

The **leap-c** software framework mentioned throughout the school implements these concepts, providing a differentiable MPC layer for PyTorch with efficient batched implementations for learning. This bridges the gap between optimization and machine learning frameworks, making research in this area more accessible.

## Challenges and Open Questions

Despite the progress, significant challenges remain. Computational efficiency limits scaling to high-dimensional offline RL with massive datasets. Risk-aware learning beyond robust MPC toward sophisticated risk metrics like mission completion probabilities remains underdeveloped. Multi-agent coordination with distributed decision-making is largely unsolved. Expanding the framework beyond MPC to general planning problems like combinatorial optimization or stochastic multi-stage programming opens new frontiers.

The interplay between safety and optimality creates inherent tensions. Hard constraints in MPC guarantee safety with respect to the model but may be overly conservative or even infeasible when the model is imperfect. Soft constraints allow constraint violation but lose formal guarantees. Learning can help by adapting constraints based on observed system behavior, but verifying the safety of learned components remains difficult, especially for neural networks.

The question of generalization is subtle. MPC naturally generalizes to new setpoints or operating conditions if the model and cost remain valid. But neural network policies trained on specific tasks may fail catastrophically when conditions change beyond their training distribution. Combining MPC's principled generalization with neural networks' representational power remains an active research area.

## Reflections and Future Directions

This fall school transformed my understanding of sequential decision-making. The separation between control and machine learning communities seemed artificial by the end—both solve MDPs but emphasize different aspects and make different computational tradeoffs.

The paradigm shift from model fitting to policy learning represents a fundamental rethinking of model-based control. We're not trying to build the most accurate simulator; we're trying to build the best decision-maker. The model, cost, and constraints are all part of a holistic representation whose purpose is producing good actions, not necessarily accurate predictions.

The theoretical results on MDP equivalences provide a solid foundation but leave practical questions. How do we choose parameterizations that are both expressive enough to represent optimal policies and structured enough for efficient learning? How do we balance the interpretability and safety guarantees of MPC with the flexibility of learned components? When should we favor aligned learning (maintaining physical meaning) versus closed-loop learning (maximizing performance)?

The connection to other areas became apparent throughout the week. The relationship between MPC and dynamic programming illuminates how receding horizon control implicitly approximates infinite-horizon optimal control. The similarity between policy gradient methods and sensitivity-based MPC tuning suggests unified optimization frameworks. The parallels between state estimation and representation learning in RL point toward integrated approaches to perception and control.

Looking forward, several directions seem particularly promising. Belief state MPC, where the state includes uncertainty estimates and the controller explicitly reasons about information gathering, could bridge partial observability and active learning. Learning world models specifically optimized for control rather than prediction accuracy aligns with the theoretical insights. Compositional approaches that combine multiple learned and model-based components with formal guarantees could scale to complex systems. Transfer learning and meta-learning could enable controllers to quickly adapt to new tasks or environments using the MPC structure as inductive bias.

## Conclusion

This fall school provided a comprehensive view of where MPC and RL intersect and how their combination can exceed what either achieves alone. The theoretical foundations are solid, the algorithms are maturing, and the software tools are becoming available. Practical applications from energy systems to robotics demonstrate the viability of these approaches.

What resonated most was the intellectual synthesis—recognizing that planning and learning, models and data, optimization and adaptation are not opposing paradigms but complementary tools. The best solutions will likely combine the strengths of both: MPC's structured approach to constraints and physical insight, and RL's ability to learn from experience and optimize closed-loop performance.

The field is clearly at an inflection point. The theoretical understanding has reached a level where we can principally design learning algorithms that preserve MPC's strengths while addressing its limitations. The computational tools are mature enough for real deployment. The applications are compelling enough to drive sustained research. The next few years will likely see these ideas move from academic research to practical systems, transforming how we design controllers for complex, uncertain, high-stakes environments.

I'm grateful to the instructors for sharing their deep expertise and to my fellow participants for stimulating discussions. This week has equipped me with both theoretical foundations and practical tools to contribute to this exciting research area. The journey from MDPs to modern RL+MPC has been challenging but deeply rewarding, opening new perspectives on how we can build intelligent systems that both learn and reason.

---

**Course Information:**
- **Institution:** University of Freiburg
- **Instructors:** Joschka Boedecker, Moritz Diehl, Sebastien Gros, Dirk Reinhardt, Jasper Hoffmann, Andrea Ghezzi
- **Duration:** October 6-10, 2025
- **Topics Covered:** MDPs, Dynamic Programming, Model Predictive Control, Reinforcement Learning, TD Methods, Actor-Critic, Policy Gradients, MPC-RL Synthesis, Imitation Learning
- <a href="https://andngdtudk.github.io/certificates/andng_mpcrt_cert.pdf" target="_blank"> Certificate of participation </a>



