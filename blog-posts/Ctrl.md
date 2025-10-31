<!-- image: https://andngdtudk.github.io/images/tsc.png -->

# Control Theory: Mathematical Frameworks for Shaping System Behavior

*Copenhagen*, 31st October 2025

In our previous post, we established the mathematical foundations for understanding how systems evolve over time. We explored state spaces, dynamic equations, stability analysis, and the geometric intuition behind system behavior. But understanding how a system behaves naturally is only half the story. The real power comes when we can **control** that behavior — when we can design inputs that guide the system toward desired outcomes despite uncertainty, disturbances, and complex dynamics.

Control theory is the mathematical discipline that addresses this challenge. It provides rigorous frameworks for designing control laws that achieve objectives like stability, optimality, robustness, and adaptability. For AI and mobility systems, these frameworks are essential. An autonomous vehicle must control its trajectory through traffic. A multi-agent system must coordinate individual behaviors to achieve collective goals. A reinforcement learning algorithm must control its policy updates to maximize long-term rewards.

This post dives deep into the mathematical foundations of control theory, from classical linear methods to modern optimal and robust approaches. We'll see how these mathematical tools directly apply to the challenges of controlling intelligent, adaptive systems in uncertain environments.

## The Control Problem: Mathematical Formulation

At its core, a control problem involves a **plant** (the system to be controlled) and a **controller** that generates inputs to achieve desired behavior. Mathematically, we have:

**Plant dynamics:**
$$\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), \mathbf{d}(t), t)$$
$$\mathbf{y}(t) = \mathbf{h}(\mathbf{x}(t), \mathbf{u}(t), t)$$

where:
- $\mathbf{x}(t) \in \mathbb{R}^n$ is the state vector
- $\mathbf{u}(t) \in \mathbb{R}^m$ is the control input
- $\mathbf{d}(t) \in \mathbb{R}^p$ represents disturbances/uncertainty  
- $\mathbf{y}(t) \in \mathbb{R}^q$ is the measured output
- $\mathbf{f}: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \times \mathbb{R} \to \mathbb{R}^n$ defines the dynamics
- $\mathbf{h}: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R} \to \mathbb{R}^q$ defines the output map

**Controller:**
$$\mathbf{u}(t) = \boldsymbol{\pi}(\mathbf{y}(t), \mathbf{r}(t), t)$$

where $\boldsymbol{\pi}$ is the **control law** or **policy**, and $\mathbf{r}(t)$ is the **reference signal** representing the desired behavior.

The **closed-loop system** combines plant and controller:
$$\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \boldsymbol{\pi}(\mathbf{h}(\mathbf{x}(t)), \mathbf{r}(t), t), \mathbf{d}(t), t)$$

The control design problem is to choose $\boldsymbol{\pi}$ such that the closed-loop system satisfies performance specifications (stability, tracking, disturbance rejection, etc.).

## Linear Feedback Control: The Foundation

### Linear Time-Invariant Systems

For linear time-invariant (LTI) plants:
$$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{E}\mathbf{d}$$
$$\mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}$$

where $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times m}$, $\mathbf{C} \in \mathbb{R}^{q \times n}$, $\mathbf{D} \in \mathbb{R}^{q \times m}$, and $\mathbf{E} \in \mathbb{R}^{n \times p}$.

### State Feedback Control

If the full state is available for feedback, we can use:
$$\mathbf{u} = -\mathbf{K}\mathbf{x} + \mathbf{N}\mathbf{r}$$

where $\mathbf{K} \in \mathbb{R}^{m \times n}$ is the **feedback gain matrix** and $\mathbf{N} \in \mathbb{R}^{m \times \ell}$ provides reference tracking (with $\mathbf{r} \in \mathbb{R}^{\ell}$).

The closed-loop system becomes:
$$\dot{\mathbf{x}} = (\mathbf{A} - \mathbf{B}\mathbf{K})\mathbf{x} + \mathbf{B}\mathbf{N}\mathbf{r} + \mathbf{E}\mathbf{d}$$

The key insight is that feedback changes the system's eigenvalues from those of $\mathbf{A}$ to those of $\mathbf{A} - \mathbf{B}\mathbf{K}$. If the system is **controllable** (i.e., $\text{rank}([\mathbf{B} \; \mathbf{A}\mathbf{B} \; \cdots \; \mathbf{A}^{n-1}\mathbf{B}]) = n$), then we can place the closed-loop poles anywhere in the complex plane by choosing $\mathbf{K}$ appropriately.

### Pole Placement and Ackermann's Formula

For single-input systems ($m = 1$), **Ackermann's formula** provides an explicit expression for the feedback gain:

$$\mathbf{K} = [0 \; 0 \; \cdots \; 0 \; 1] \begin{bmatrix} \mathbf{B} & \mathbf{A}\mathbf{B} & \cdots & \mathbf{A}^{n-1}\mathbf{B} \end{bmatrix}^{-1} \alpha_c(\mathbf{A})$$

where $\alpha_c(\mathbf{A})$ is the **characteristic polynomial** of the desired closed-loop system evaluated at matrix $\mathbf{A}$.

If we want closed-loop poles at $\{s_1, s_2, \ldots, s_n\}$, then:
$$\alpha_c(s) = (s - s_1)(s - s_2) \cdots (s - s_n)$$

### Observer Design and the Separation Principle

Often, we cannot measure the full state directly. In this case, we design an **observer** (or **estimator**) to reconstruct the state from output measurements:

$$\dot{\hat{\mathbf{x}}} = \mathbf{A}\hat{\mathbf{x}} + \mathbf{B}\mathbf{u} + \mathbf{L}(\mathbf{y} - \mathbf{C}\hat{\mathbf{x}})$$

where $\hat{\mathbf{x}}$ is the state estimate and $\mathbf{L} \in \mathbb{R}^{n \times q}$ is the **observer gain matrix**.

The estimation error $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$ evolves according to:
$$\dot{\mathbf{e}} = (\mathbf{A} - \mathbf{L}\mathbf{C})\mathbf{e}$$

If the system is **observable**, we can choose $\mathbf{L}$ to make $\mathbf{A} - \mathbf{L}\mathbf{C}$ stable, ensuring $\mathbf{e}(t) \to \mathbf{0}$.

The **separation principle** states that we can design the controller and observer independently: use $\mathbf{u} = -\mathbf{K}\hat{\mathbf{x}} + \mathbf{N}\mathbf{r}$ where $\mathbf{K}$ is designed assuming full state feedback, and $\mathbf{L}$ is designed for fast, stable estimation.

## Optimal Control Theory: The Calculus of Variations Approach

Linear feedback control provides stability and desired transient response, but it doesn't address optimality. **Optimal control theory** formulates control as an optimization problem over function spaces.

### The General Optimal Control Problem

Consider the cost functional:
$$J[\mathbf{u}(\cdot)] = \int_0^T L(\mathbf{x}(t), \mathbf{u}(t), t) dt + \Phi(\mathbf{x}(T))$$

subject to:
$$\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t), \quad \mathbf{x}(0) = \mathbf{x}_0$$

where $L$ is the **running cost** and $\Phi$ is the **terminal cost**.

The optimal control problem is:
$$\mathbf{u}^*(t) = \arg\min_{\mathbf{u}(\cdot)} J[\mathbf{u}(\cdot)]$$

### Pontryagin's Maximum Principle

The **Hamiltonian** is defined as:
$$H(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}, t) = L(\mathbf{x}, \mathbf{u}, t) + \boldsymbol{\lambda}^T \mathbf{f}(\mathbf{x}, \mathbf{u}, t)$$

where $\boldsymbol{\lambda}(t) \in \mathbb{R}^n$ is the **costate** or **adjoint variable**.

**Pontryagin's Maximum Principle** provides necessary conditions for optimality:

1. **State equation**: $\dot{\mathbf{x}} = \frac{\partial H}{\partial \boldsymbol{\lambda}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t)$

2. **Costate equation**: $\dot{\boldsymbol{\lambda}} = -\frac{\partial H}{\partial \mathbf{x}}$

3. **Stationarity condition**: $\frac{\partial H}{\partial \mathbf{u}} = 0$ (if $\mathbf{u}$ is unconstrained)

4. **Transversality condition**: $\boldsymbol{\lambda}(T) = \frac{\partial \Phi}{\partial \mathbf{x}}\bigg|_{\mathbf{x}(T)}$

These conditions define a **two-point boundary value problem** (TPBVP) that must be solved to find the optimal trajectory.

### Linear Quadratic Regulator (LQR)

For linear systems with quadratic costs:
$$J = \int_0^T (\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{u}^T\mathbf{R}\mathbf{u}) dt + \mathbf{x}(T)^T\mathbf{P}_f\mathbf{x}(T)$$

where $\mathbf{Q} \succeq 0$, $\mathbf{R} \succ 0$, and $\mathbf{P}_f \succeq 0$.

The optimal control has the form:
$$\mathbf{u}^*(t) = -\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P}(t)\mathbf{x}(t)$$

where $\mathbf{P}(t)$ satisfies the **matrix Riccati differential equation**:
$$-\dot{\mathbf{P}} = \mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A} - \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P} + \mathbf{Q}$$

with terminal condition $\mathbf{P}(T) = \mathbf{P}_f$.

For the infinite-horizon case ($T \to \infty$), $\mathbf{P}$ converges to the solution of the **algebraic Riccati equation (ARE)**:
$$\mathbf{0} = \mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A} - \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P} + \mathbf{Q}$$

The optimal feedback gain is then:
$$\mathbf{K} = \mathbf{R}^{-1}\mathbf{B}^T\mathbf{P}$$

### Hamilton-Jacobi-Bellman (HJB) Equation

For nonlinear systems, **dynamic programming** leads to the **Hamilton-Jacobi-Bellman equation**:

$$\frac{\partial V}{\partial t} + \min_{\mathbf{u}} \left[ L(\mathbf{x}, \mathbf{u}) + \left(\frac{\partial V}{\partial \mathbf{x}}\right)^T \mathbf{f}(\mathbf{x}, \mathbf{u}) \right] = 0$$

with terminal condition $V(\mathbf{x}, T) = \Phi(\mathbf{x})$.

Here, $V(\mathbf{x}, t)$ is the **value function** — the minimum cost-to-go from state $\mathbf{x}$ at time $t$.

If $V$ is smooth, the optimal control is:
$$\mathbf{u}^*(\mathbf{x}, t) = \arg\min_{\mathbf{u}} \left[ L(\mathbf{x}, \mathbf{u}) + \left(\frac{\partial V}{\partial \mathbf{x}}\right)^T \mathbf{f}(\mathbf{x}, \mathbf{u}) \right]$$

The HJB equation is generally impossible to solve analytically for nonlinear systems, but it provides the theoretical foundation for numerical methods and approximate approaches.

## Model Predictive Control (MPC)

**Model Predictive Control** combines the optimality of optimal control with the practicality of receding horizon implementation.

### The MPC Algorithm

At each time $t$, solve the finite-horizon optimal control problem:

$$\min_{\{\mathbf{u}(\tau)\}_{\tau=t}^{t+N-1}} \sum_{k=0}^{N-1} L(\mathbf{x}(t+k|t), \mathbf{u}(t+k|t)) + \Phi(\mathbf{x}(t+N|t))$$

subject to:
$$\mathbf{x}(t+k+1|t) = \mathbf{f}(\mathbf{x}(t+k|t), \mathbf{u}(t+k|t))$$
$$\mathbf{x}(t|t) = \mathbf{x}(t)$$ (current state)
$$\mathbf{u}(t+k|t) \in \mathcal{U}, \quad \mathbf{x}(t+k|t) \in \mathcal{X}$$ (constraints)

Apply only the first control action $\mathbf{u}^*(t) = \mathbf{u}^*(t|t)$, then repeat at time $t+1$ with updated state information.

### Stability of MPC

For nonlinear MPC, stability requires careful design of the terminal cost $\Phi$ and terminal constraint set. A common approach is to choose:

1. **Terminal constraint**: $\mathbf{x}(t+N|t) \in \mathcal{X}_f$ where $\mathcal{X}_f$ is a **controlled invariant set**
2. **Terminal cost**: $\Phi(\mathbf{x}) = V_f(\mathbf{x})$ where $V_f$ is a **control Lyapunov function** on $\mathcal{X}_f$

If these conditions are satisfied, the MPC law guarantees closed-loop stability.

### Linear MPC (LMPC)

For linear systems with convex constraints, the MPC problem becomes a **quadratic program** (QP):

$$\min_{\mathbf{U}} \mathbf{U}^T\mathbf{H}\mathbf{U} + \mathbf{f}^T\mathbf{U}$$
subject to: $\mathbf{A}_{\text{ineq}}\mathbf{U} \leq \mathbf{b}$, $\mathbf{A}_{\text{eq}}\mathbf{U} = \mathbf{c}$

where $\mathbf{U} = [\mathbf{u}(t|t)^T \; \mathbf{u}(t+1|t)^T \; \cdots \; \mathbf{u}(t+N-1|t)^T]^T$ is the decision variable.

The matrices $\mathbf{H}$, $\mathbf{f}$, $\mathbf{A}_{\text{ineq}}$, $\mathbf{A}_{\text{eq}}$, $\mathbf{b}$, $\mathbf{c}$ encode the cost function, dynamics, and constraints. QPs can be solved efficiently using interior-point or active-set methods.

## Robust Control Theory

Real systems are never perfectly modeled. **Robust control** addresses performance degradation due to model uncertainty.

### Uncertainty Models

**Parametric uncertainty**: The system matrices are not exactly known:
$$\mathbf{A} = \mathbf{A}_0 + \delta\mathbf{A}, \quad \mathbf{B} = \mathbf{B}_0 + \delta\mathbf{B}$$

where $\mathbf{A}_0$, $\mathbf{B}_0$ are nominal values and $\delta\mathbf{A}$, $\delta\mathbf{B}$ represent uncertainty.

**Additive uncertainty**: 
$$G(s) = G_0(s) + \Delta(s)$$

**Multiplicative uncertainty**:
$$G(s) = G_0(s)(1 + \Delta(s))$$

where $G_0(s)$ is the nominal transfer function and $\Delta(s)$ represents uncertainty with $\|\Delta(s)\|_{\infty} \leq \gamma$ for some $\gamma > 0$.

### H∞ Control

**$H_{\infty}$ control** minimizes the worst-case energy gain from disturbances to performance outputs.

For the system:
$$\begin{bmatrix} \dot{\mathbf{x}} \\ \mathbf{z} \\ \mathbf{y} \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{B}_1 & \mathbf{B}_2 \\ \mathbf{C}_1 & \mathbf{D}_{11} & \mathbf{D}_{12} \\ \mathbf{C}_2 & \mathbf{D}_{21} & \mathbf{D}_{22} \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \mathbf{w} \\ \mathbf{u} \end{bmatrix}$$

where $\mathbf{w}$ is the disturbance input, $\mathbf{z}$ is the performance output, and $\mathbf{y}$ is the measured output.

The $H_{\infty}$ norm of the closed-loop transfer function from $\mathbf{w}$ to $\mathbf{z}$ is:
$$\|T_{zw}\|_{\infty} = \sup_{\omega \in \mathbb{R}} \bar{\sigma}(T_{zw}(j\omega))$$

where $\bar{\sigma}(\cdot)$ is the largest singular value.

The **$H_{\infty}$ optimal control problem** is:
$$\min_{\mathbf{K}} \|T_{zw}\|_{\infty}$$

subject to closed-loop stability.

### Riccati-Based Solution

For the standard $H_{\infty}$ problem, the optimal controller exists if and only if:

1. $(\mathbf{A}, \mathbf{B}_2)$ is stabilizable and $(\mathbf{C}_2, \mathbf{A})$ is detectable
2. The control Riccati equation has a stabilizing solution $\mathbf{P} \geq 0$:
   $$\mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A} + \mathbf{P}(\gamma^{-2}\mathbf{B}_1\mathbf{B}_1^T - \mathbf{B}_2\mathbf{B}_2^T)\mathbf{P} + \mathbf{C}_1^T\mathbf{C}_1 = \mathbf{0}$$

3. The filter Riccati equation has a stabilizing solution $\mathbf{Q} \geq 0$:
   $$\mathbf{A}\mathbf{Q} + \mathbf{Q}\mathbf{A}^T + \mathbf{Q}(\gamma^{-2}\mathbf{C}_1^T\mathbf{C}_1 - \mathbf{C}_2^T\mathbf{C}_2)\mathbf{Q} + \mathbf{B}_1\mathbf{B}_1^T = \mathbf{0}$$

4. $\rho(\mathbf{P}\mathbf{Q}) < \gamma^2$ (spectral radius condition)

## Adaptive and Learning-Based Control

Traditional control assumes fixed system models. **Adaptive control** handles parametric uncertainty by estimating unknown parameters online.

### Model Reference Adaptive Control (MRAC)

Consider a plant with unknown parameters:
$$\dot{\mathbf{x}}_p = \mathbf{A}_p\mathbf{x}_p + \mathbf{B}_p\mathbf{u}$$

We want the plant to follow a reference model:
$$\dot{\mathbf{x}}_m = \mathbf{A}_m\mathbf{x}_m + \mathbf{B}_m\mathbf{r}$$

Define the tracking error $\mathbf{e} = \mathbf{x}_p - \mathbf{x}_m$ and use the control law:
$$\mathbf{u} = \boldsymbol{\Theta}_x^T(t)\mathbf{x}_p + \boldsymbol{\Theta}_r^T(t)\mathbf{r}$$

The adaptation laws update the controller parameters:
$$\dot{\boldsymbol{\Theta}}_x = -\boldsymbol{\Gamma}_x \mathbf{x}_p \mathbf{e}^T \mathbf{P} \mathbf{B}_p$$
$$\dot{\boldsymbol{\Theta}}_r = -\boldsymbol{\Gamma}_r \mathbf{r} \mathbf{e}^T \mathbf{P} \mathbf{B}_p$$

where $\boldsymbol{\Gamma}_x, \boldsymbol{\Gamma}_r > 0$ are adaptation gains and $\mathbf{P} > 0$ satisfies $\mathbf{A}_m^T \mathbf{P} + \mathbf{P} \mathbf{A}_m = -\mathbf{Q}$ for some $\mathbf{Q} > 0$.

### Reinforcement Learning and Control

**Reinforcement Learning** can be viewed as adaptive optimal control for unknown systems. The **policy gradient theorem** shows that:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\pi}_{\boldsymbol{\theta}}} \left[ \sum_{t=0}^{\infty} \gamma^t \nabla_{\boldsymbol{\theta}} \log \boldsymbol{\pi}_{\boldsymbol{\theta}}(\mathbf{u}_t | \mathbf{x}_t) Q^{\boldsymbol{\pi}_{\boldsymbol{\theta}}}(\mathbf{x}_t, \mathbf{u}_t) \right]$$

where $J(\boldsymbol{\theta})$ is the expected return, $\boldsymbol{\pi}_{\boldsymbol{\theta}}$ is a parameterized policy, and $Q^{\boldsymbol{\pi}_{\boldsymbol{\theta}}}$ is the action-value function.

This connects RL to optimal control: the policy gradient algorithm performs gradient ascent in policy space, similar to how classical optimal control varies the control function to minimize cost.

## Applications to AI and Mobility Systems

These mathematical frameworks directly enable intelligent control systems:

### Autonomous Vehicle Control

**Longitudinal control** (speed regulation):
- **Plant**: Vehicle dynamics $\dot{v} = \frac{1}{m}(F_{\text{drive}} - F_{\text{drag}} - mg\sin\theta)$
- **Controller**: MPC with constraints on acceleration, comfort, and safety gaps
- **Disturbances**: Grade changes, wind, traffic interactions

**Lateral control** (path following):
- **Plant**: Bicycle model with front-wheel steering
- **Controller**: LQR or $H_{\infty}$ for robustness to road variations
- **Reference**: Planned trajectory from path planning algorithm

### Multi-Agent Coordination

**Consensus control**: Each agent $i$ has dynamics:
$$\dot{\mathbf{x}}_i = \mathbf{A}\mathbf{x}_i + \mathbf{B}\mathbf{u}_i$$

The control law:
$$\mathbf{u}_i = \mathbf{K} \sum_{j \in \mathcal{N}_i} a_{ij}(\mathbf{x}_j - \mathbf{x}_i)$$

achieves consensus (all states converge to the same value) if the communication graph is connected and $\mathbf{K}$ is designed appropriately.

### Traffic Signal Control

**Plant**: Traffic flow dynamics (macroscopic models)
**Controller**: MPC with switching constraints and discrete signal phases
**Objective**: Minimize total delay while maintaining safety and fairness

## The Bridge to Modern AI

The mathematical frameworks we've explored provide the foundation for understanding how modern AI systems control complex, uncertain environments:

- **Deep Reinforcement Learning** extends classical optimal control to high-dimensional, unknown systems using neural network approximations
- **Model Predictive Control** provides the theoretical basis for planning algorithms in robotics and autonomous systems  
- **Robust control theory** informs the design of AI systems that must perform reliably despite model uncertainty and adversarial inputs
- **Adaptive control** principles guide how AI systems should update their behaviors based on new experience

In our next post, we'll explore **optimization theory** — the mathematical foundation underlying both classical control design and modern machine learning. We'll see how convex optimization, gradient methods, and constrained optimization connect control theory to the algorithms that power today's intelligent systems.

The journey from classical control to modern AI isn't a replacement of old methods with new ones — it's a deepening understanding of how mathematical principles can be extended, generalized, and combined to address increasingly complex challenges. The autonomous vehicles of tomorrow will use all of these tools: classical control for low-level stability, optimal control for trajectory planning, robust control for handling uncertainty, adaptive control for learning and improvement, and AI for high-level decision-making in complex environments.

---

**References**

**Åström, K. J., & Murray, R. M.** (2021). *Feedback systems: An introduction for scientists and engineers* (2nd ed.). Princeton University Press.

**Boyd, S., & Barratt, C.** (1991). *Linear controller design: Limits of performance*. Prentice Hall.

**Rawlings, J. B., Mayne, D. Q., & Diehl, M.** (2017). *Model predictive control: Theory, computation, and design* (2nd ed.). Nob Hill Publishing.

**Zhou, K., & Doyle, J. C.** (1998). *Essentials of robust control*. Prentice Hall.

**Ioannou, P. A., & Sun, J.** (1996). *Robust adaptive control*. Prentice Hall.

**Kirk, D. E.** (2004). *Optimal control theory: An introduction*. Dover Publications.

**Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
