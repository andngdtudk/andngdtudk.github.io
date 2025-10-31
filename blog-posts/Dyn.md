<!-- image: https://andngdtudk.github.io/images/radio_cambridge.jpg -->
# Dynamic Systems: Mathematical Foundations for Time-Evolving Behavior

*Copenhagen*, 10th October 2025

In our previous post, we established that systems are more than collections of parts — they are organized wholes that exhibit emergent behaviors through the interactions of their components. But there's a crucial dimension we've only touched upon: **time**. The most interesting and challenging systems are those that evolve, adapt, and change over time. These are **dynamic systems**, and understanding them requires mathematical precision.

Consider an autonomous vehicle navigating through traffic. Its position, velocity, and internal decision state are constantly changing in response to sensor inputs, other vehicles, and its own learning algorithms. A logistics network continuously adapts routing decisions based on demand fluctuations, capacity constraints, and real-time disruptions. A multi-agent reinforcement learning system evolves its policies through experience, with each agent's behavior influencing others in complex feedback loops.

What unites these examples is that their behavior unfolds over time according to mathematical rules — rules we can model, analyze, and sometimes control. This post dives deep into the mathematical foundations that make such analysis possible.

## The State Space: A Mathematical Universe

At the heart of dynamic systems theory lies the concept of **state**. The state of a system at any given time is a complete description of all the information needed to predict its future behavior (given knowledge of future inputs and the system's dynamics).

Mathematically, we represent the state as a vector $\mathbf{x}(t) \in \mathbb{R}^n$, where each component captures a different aspect of the system's condition:

$$\mathbf{x}(t) = \begin{bmatrix} x_1(t) \\ x_2(t) \\ \vdots \\ x_n(t) \end{bmatrix}$$

The **state space** $\mathcal{X}$ is the set of all possible states the system can occupy. For a simple autonomous vehicle, the state might include:

$$\mathbf{x}(t) = \begin{bmatrix} \text{position}_x(t) \\ \text{position}_y(t) \\ \text{velocity}_x(t) \\ \text{velocity}_y(t) \\ \text{heading}(t) \end{bmatrix} \in \mathbb{R}^5$$

But for a more sophisticated system — say, a learning agent with internal memory and decision-making processes — the state space might be much higher dimensional and include discrete components representing the agent's current policy parameters, memory states, or decision modes.

The power of the state space representation is that it provides a geometric view of system behavior. Every possible system configuration corresponds to a point in this space, and the system's evolution over time traces out a **trajectory** or **orbit** through the state space.

## System Dynamics: The Rules of Evolution

The evolution of a dynamic system is governed by its **dynamics** — mathematical rules that specify how the state changes over time. We can represent these rules in several ways:

### Continuous-Time Systems

For systems that evolve continuously, we use **differential equations**:

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t)$$

where:
- $\mathbf{x}(t) \in \mathbb{R}^n$ is the state vector at time $t$
- $\mathbf{u}(t) \in \mathbb{R}^m$ is the control input (if the system is controllable)
- $\mathbf{f}: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R} \to \mathbb{R}^n$ is the **vector field** defining the dynamics

The vector field $\mathbf{f}$ assigns to each point in state space and each time $t$ a velocity vector indicating the direction and speed of state evolution at that point.

For a double integrator system (like a vehicle with controllable acceleration), the dynamics might be:

$$\frac{d}{dt}\begin{bmatrix} p \\ v \end{bmatrix} = \begin{bmatrix} v \\ u \end{bmatrix}$$

where $p$ is position, $v$ is velocity, and $u$ is the acceleration control input.

### Discrete-Time Systems  

Many AI and computational systems are more naturally described in discrete time steps:

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, k)$$

where $k \in \mathbb{N}$ represents the time step. This formulation is particularly natural for reinforcement learning systems, where an agent observes state $\mathbf{x}_k$, takes action $\mathbf{u}_k$, and transitions to state $\mathbf{x}_{k+1}$ according to environment dynamics.

### Stochastic Systems

Real-world systems almost always involve uncertainty. We can incorporate this through **stochastic differential equations** (SDEs):

$$d\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t)dt + \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t), t)d\mathbf{W}(t)$$

where $\mathbf{W}(t)$ is a Wiener process (Brownian motion) and $\mathbf{g}$ determines how noise affects the system.

For discrete-time systems, we often use:

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, k) + \mathbf{w}_k$$

where $\mathbf{w}_k$ is a random disturbance, typically assumed to be white noise with known statistics.

## Linear vs. Nonlinear Dynamics

The mathematical structure of the dynamics function $\mathbf{f}$ fundamentally determines what analysis tools we can apply and what behaviors we can expect.

### Linear Time-Invariant (LTI) Systems

The simplest class of dynamic systems has linear dynamics:

$$\frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)$$

where $\mathbf{A} \in \mathbb{R}^{n \times n}$ is the **system matrix** and $\mathbf{B} \in \mathbb{R}^{n \times m}$ is the **input matrix**.

The beauty of linear systems is that they have closed-form solutions:

$$\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}(0) + \int_0^t e^{\mathbf{A}(t-\tau)}\mathbf{B}\mathbf{u}(\tau)d\tau$$

where $e^{\mathbf{A}t}$ is the **matrix exponential**, representing the system's natural evolution from initial condition $\mathbf{x}(0)$.

The eigenvalues $\lambda_i$ of matrix $\mathbf{A}$ completely characterize the system's stability:
- If $\text{Re}(\lambda_i) < 0$ for all $i$, the system is **asymptotically stable**
- If any $\text{Re}(\lambda_i) > 0$, the system is **unstable**  
- If some $\text{Re}(\lambda_i) = 0$ and others are negative, the system is **marginally stable**

### Nonlinear Systems

Most real-world systems, especially those involving AI and complex interactions, are nonlinear:

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t))$$

where $\mathbf{f}$ is a nonlinear function. Nonlinear systems can exhibit rich behaviors impossible in linear systems:

- **Multiple equilibria**: Several stable operating points
- **Limit cycles**: Periodic orbits that attract nearby trajectories
- **Chaos**: Sensitive dependence on initial conditions
- **Bifurcations**: Qualitative changes in behavior as parameters vary

For example, consider the nonlinear dynamics of a pendulum:

$$\frac{d}{dt}\begin{bmatrix} \theta \\ \dot{\theta} \end{bmatrix} = \begin{bmatrix} \dot{\theta} \\ -\frac{g}{l}\sin(\theta) - \frac{b}{m}\dot{\theta} + \frac{1}{ml^2}u \end{bmatrix}$$

The $\sin(\theta)$ term makes this system nonlinear and gives it fundamentally different behavior near different equilibria.

## Stability and Equilibria

Understanding the long-term behavior of dynamic systems requires analyzing their **equilibria** — states where the system would remain if left undisturbed.

### Equilibrium Points

An equilibrium point $\mathbf{x}^*$ satisfies:

$$\mathbf{f}(\mathbf{x}^*, \mathbf{0}) = \mathbf{0}$$

For autonomous vehicles, an equilibrium might represent maintaining constant velocity in a straight line. For a multi-agent system, it might represent a Nash equilibrium where no agent can improve by unilaterally changing strategy.

### Lyapunov Stability

For nonlinear systems, we analyze stability using **Lyapunov theory**. An equilibrium $\mathbf{x}^*$ is **Lyapunov stable** if for every $\epsilon > 0$, there exists $\delta > 0$ such that:

$$\|\mathbf{x}(0) - \mathbf{x}^*\| < \delta \implies \|\mathbf{x}(t) - \mathbf{x}^*\| < \epsilon \quad \forall t \geq 0$$

It is **asymptotically stable** if it is Lyapunov stable and trajectories starting near $\mathbf{x}^*$ actually converge to it:

$$\lim_{t \to \infty} \mathbf{x}(t) = \mathbf{x}^*$$

### Lyapunov Functions

To prove stability, we seek a **Lyapunov function** $V(\mathbf{x})$ that satisfies:
1. $V(\mathbf{x}) > 0$ for $\mathbf{x} \neq \mathbf{x}^*$ (positive definite)
2. $V(\mathbf{x}^*) = 0$
3. $\dot{V}(\mathbf{x}) = \nabla V \cdot \mathbf{f}(\mathbf{x}) \leq 0$ along trajectories

If such a function exists, the equilibrium is stable. If $\dot{V}(\mathbf{x}) < 0$ (except at $\mathbf{x}^*$), it's asymptotically stable.

## Observability and Controllability

For systems we want to monitor and control, two key mathematical properties determine what's possible:

### Controllability

A system is **controllable** if we can steer it from any initial state to any desired final state using appropriate control inputs.

For linear systems $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$, controllability is determined by the **controllability matrix**:

$$\mathcal{C} = \begin{bmatrix} \mathbf{B} & \mathbf{A}\mathbf{B} & \mathbf{A}^2\mathbf{B} & \cdots & \mathbf{A}^{n-1}\mathbf{B} \end{bmatrix}$$

The system is controllable if and only if $\text{rank}(\mathcal{C}) = n$.

### Observability

A system is **observable** if we can determine its internal state from measurements of its outputs.

For systems with output equation $\mathbf{y} = \mathbf{C}\mathbf{x}$, observability is determined by the **observability matrix**:

$$\mathcal{O} = \begin{bmatrix} \mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \mathbf{C}\mathbf{A}^2 \\ \vdots \\ \mathbf{C}\mathbf{A}^{n-1} \end{bmatrix}$$

The system is observable if and only if $\text{rank}(\mathcal{O}) = n$.

These concepts are crucial for AI systems. An autonomous vehicle's control system must be controllable (we can influence its behavior through actuator commands) and its perception system must provide observable state information (we can estimate the vehicle's state from sensor measurements).

## Phase Portraits and Geometric Intuition

One of the most powerful tools for understanding dynamic systems is the **phase portrait** — a geometric visualization of trajectories in state space.

Consider a simple predator-prey system:

$$\begin{align}
\frac{dx}{dt} &= ax - bxy \\
\frac{dy}{dt} &= -cy + dxy
\end{align}$$

where $x$ represents prey population and $y$ represents predator population. The phase portrait reveals closed orbits around the equilibrium point $(\frac{c}{d}, \frac{a}{b})$, indicating periodic oscillations.

For AI applications, phase portraits help visualize:
- **Convergence patterns** in learning algorithms
- **Basin of attraction** for different equilibria  
- **Separatrices** that divide different behavioral regions
- **Limit cycles** representing periodic behaviors

## Linearization and Local Analysis

For nonlinear systems, we often gain insight through **linearization** around equilibria. Given a nonlinear system:

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$$

We can approximate it near equilibrium $\mathbf{x}^*$ using the **Jacobian matrix**:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}=\mathbf{x}^*}$$

The linearized system becomes:

$$\frac{d(\delta\mathbf{x})}{dt} = \mathbf{J}\delta\mathbf{x}$$

where $\delta\mathbf{x} = \mathbf{x} - \mathbf{x}^*$ is the deviation from equilibrium.

The eigenvalues of $\mathbf{J}$ determine local stability properties, and the eigenvectors indicate the principal directions of evolution near the equilibrium.

## Applications to AI and Mobility Systems

These mathematical concepts directly apply to modern AI and mobility challenges:

### Reinforcement Learning Dynamics

Policy gradient methods can be viewed as dynamic systems where the policy parameters evolve according to:

$$\frac{d\boldsymbol{\theta}}{dt} = \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

where $J(\boldsymbol{\theta})$ is the expected return. The convergence properties depend on the landscape of $J$ and can be analyzed using dynamical systems tools.

### Multi-Agent Coordination

Multi-agent systems often have dynamics of the form:

$$\dot{\mathbf{x}}_i = \mathbf{f}_i(\mathbf{x}_1, \ldots, \mathbf{x}_N, \mathbf{u}_i)$$

where each agent $i$'s state evolution depends on all other agents' states. Nash equilibria correspond to equilibrium points of the coupled system.

### Traffic Flow Dynamics

Macroscopic traffic models use partial differential equations, but when discretized spatially, they become high-dimensional ordinary differential equation systems describing traffic density and flow evolution.

## Looking Forward: Control and Optimization

Understanding dynamic systems is just the beginning. In our next post, we'll explore how to **control** these systems — how to design inputs $\mathbf{u}(t)$ that shape system behavior to achieve desired objectives. We'll cover optimal control theory, model predictive control, and the connections to reinforcement learning.

The mathematical foundations we've established here — state spaces, dynamics, stability, controllability, and observability — provide the language for formulating and solving control problems. They're the building blocks for everything from autonomous vehicle path planning to multi-agent coordination to adaptive traffic management systems.

But before we move on to control, it's worth appreciating the profound insight that dynamic systems theory provides: complex, time-evolving behavior can be understood through mathematical precision. The seemingly chaotic dance of traffic, the emergence of coordination in robot swarms, the convergence of learning algorithms — all of these can be analyzed using the tools we've developed here.

This mathematical lens doesn't just help us understand existing systems; it guides us in designing new ones. When we know the mathematical conditions for stability, we can engineer systems that are inherently robust. When we understand controllability, we can ensure our designs are actually steerable. When we grasp observability, we can build systems that are properly instrumented for monitoring and adaptation.

In the next post, we'll see how these insights translate into practical control strategies for the complex, uncertain, multi-agent world of modern AI and mobility systems.

---

**References**

**Khalil, H. K.** (2002). *Nonlinear systems* (3rd ed.). Prentice Hall.

**Strogatz, S. H.** (2014). *Nonlinear dynamics and chaos: With applications to physics, biology, chemistry, and engineering* (2nd ed.). Westview Press.

**Chen, C. T.** (1999). *Linear system theory and design* (3rd ed.). Oxford University Press.

**Sastry, S.** (1999). *Nonlinear systems: Analysis, stability, and control*. Springer.

**Liberzon, D.** (2011). *Calculus of variations and optimal control theory: A concise introduction*. Princeton University Press.
