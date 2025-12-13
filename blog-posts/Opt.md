# Optimization Theory: The Mathematical Foundation of Intelligent Decision-Making

*Copenhagen*, 12th December 2025

In our previous post, we explored control theory, the mathematical framework for designing inputs that guide dynamic systems toward desired behaviors. We saw how optimal control problems, from LQR to MPC to $H \infty$ control, fundamentally involve minimizing cost functions subject to constraints. But we treated optimization itself as a tool, a means to an end. In this post, we dive deeper into optimization theory itself, revealing it as the unifying mathematical language underlying not just control, but also machine learning, operations research, and artificial intelligence.

Every intelligent system, whether classical or AI-based, fundamentally solves optimization problems. An autonomous vehicle optimizes its trajectory through traffic. A neural network optimizes its weights to minimize prediction error. A logistics network optimizes routing decisions to minimize cost while satisfying delivery constraints. A reinforcement learning agent optimizes its policy to maximize long-term rewards. Understanding the mathematical foundations of optimization is essential for anyone working at the intersection of AI, control, and complex systems.

This post explores the rich mathematical theory of optimization, from the geometric insights of convexity to the algorithmic foundations of gradient methods, from constrained optimization and duality theory to stochastic and non-convex optimization that power modern deep learning.

## The Optimization Problem: Mathematical Structure

An optimization problem in its most general form seeks to find:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{D}} f(\mathbf{x})$$

subject to:
$$g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \quad \text{(inequality constraints)}$$
$$h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \quad \text{(equality constraints)}$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ is the **decision variable** or **optimization variable**
- $f: \mathbb{R}^n \to \mathbb{R}$ is the **objective function** or **cost function**
- $g_i: \mathbb{R}^n \to \mathbb{R}$ are inequality constraint functions
- $h_j: \mathbb{R}^n \to \mathbb{R}$ are equality constraint functions
- $\mathcal{D} \subseteq \mathbb{R}^n$ is the **domain** of the problem

The **feasible set** is:
$$\mathcal{F} = \{\mathbf{x} \in \mathcal{D} : g_i(\mathbf{x}) \leq 0, \; h_j(\mathbf{x}) = 0 \; \forall i, j\}$$

A point $\mathbf{x}^*$ is a **global minimum** if $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{F}$.

A point $\mathbf{x}^*$ is a **local minimum** if there exists $\epsilon > 0$ such that $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{F}$ with $\|\mathbf{x} - \mathbf{x}^*\| < \epsilon$.

## Convex Optimization: The Tractable Core

Convex optimization problems have special structure that makes them tractable both theoretically and computationally. They form the foundation for many practical algorithms in control, machine learning, and operations research.

### Convex Sets

A set $\mathcal{C} \subseteq \mathbb{R}^n$ is **convex** if for any $\mathbf{x}_1, \mathbf{x}_2 \in \mathcal{C}$ and $\theta \in [0, 1]$:
$$\theta \mathbf{x}_1 + (1 - \theta)\mathbf{x}_2 \in \mathcal{C}$$

Geometrically, this means the line segment connecting any two points in the set lies entirely within the set.

**Examples of convex sets**:
- Hyperplanes: $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} = b\}$
- Half-spaces: $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} \leq b\}$
- Norm balls: $\{\mathbf{x} : \|\mathbf{x} - \mathbf{x}_c\| \leq r\}$
- Polyhedra: $\{\mathbf{x} : \mathbf{A}\mathbf{x} \leq \mathbf{b}, \mathbf{C}\mathbf{x} = \mathbf{d}\}$
- Ellipsoids: $\{\mathbf{x} : (\mathbf{x} - \mathbf{x}_c)^T\mathbf{P}^{-1}(\mathbf{x} - \mathbf{x}_c) \leq 1\}$ for $\mathbf{P} \succ 0$

**Operations preserving convexity**:
- Intersection: If $\mathcal{C}_1, \mathcal{C}_2$ are convex, then $\mathcal{C}_1 \cap \mathcal{C}_2$ is convex
- Affine transformation: If $\mathcal{C}$ is convex, then $\{\mathbf{A}\mathbf{x} + \mathbf{b} : \mathbf{x} \in \mathcal{C}\}$ is convex
- Cartesian product: If $\mathcal{C}_1, \mathcal{C}_2$ are convex, then $\mathcal{C}_1 \times \mathcf{C}_2$ is convex

### Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain is convex and for all $\mathbf{x}_1, \mathbf{x}_2$ in the domain and $\theta \in [0, 1]$:
$$f(\theta \mathbf{x}_1 + (1 - \theta)\mathbf{x}_2) \leq \theta f(\mathbf{x}_1) + (1 - \theta) f(\mathbf{x}_2)$$

Geometrically, the line segment between any two points on the graph of $f$ lies above the graph.

**First-order condition**: If $f$ is differentiable, it is convex if and only if:
$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) \quad \forall \mathbf{x}, \mathbf{y}$$

This says the first-order Taylor approximation is a global underestimator.

**Second-order condition**: If $f$ is twice differentiable, it is convex if and only if the **Hessian** $\nabla^2 f(\mathbf{x})$ is positive semidefinite for all $\mathbf{x}$:
$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

**Examples of convex functions**:
- Linear: $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} + b$
- Quadratic: $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x}$ for $\mathbf{Q} \succeq 0$
- Norms: $f(\mathbf{x}) = \|\mathbf{x}\|$ for any norm
- Exponential: $f(x) = e^{ax}$ for any $a$
- Log-sum-exp: $f(\mathbf{x}) = \log(\sum_{i=1}^n e^{x_i})$

**Operations preserving convexity**:
- Non-negative weighted sum: If $f_1, \ldots, f_k$ are convex and $w_i \geq 0$, then $\sum_{i=1}^k w_i f_i$ is convex
- Composition with affine: If $f$ is convex, then $g(\mathbf{x}) = f(\mathbf{A}\mathbf{x} + \mathbf{b})$ is convex
- Pointwise maximum: If $f_1, \ldots, f_k$ are convex, then $f(\mathbf{x}) = \max_i f_i(\mathbf{x})$ is convex

### Why Convexity Matters

**Fundamental property**: For convex optimization problems, **any local minimum is a global minimum**.

**Proof sketch**: Suppose $\mathbf{x}^*$ is a local minimum but not global. Then there exists $\mathbf{y} \in \mathcal{F}$ with $f(\mathbf{y}) < f(\mathbf{x}^*)$. By convexity of the feasible set, the line segment $\mathbf{x}(\theta) = \theta\mathbf{y} + (1-\theta)\mathbf{x}^*$ lies in $\mathcal{F}$ for $\theta \in [0,1]$. By convexity of $f$:
$$f(\mathbf{x}(\theta)) \leq \theta f(\mathbf{y}) + (1-\theta)f(\mathbf{x}^*) < f(\mathbf{x}^*)$$
for small $\theta > 0$, contradicting local minimality. □

This property means we can use local search algorithms with confidence that they will find the global optimum.

## Optimality Conditions

### Unconstrained Optimization

For unconstrained problems $\min_{\mathbf{x}} f(\mathbf{x})$, the **first-order necessary condition** for a local minimum is:
$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

Such points are called **stationary points** or **critical points**.

The **second-order necessary condition** is:
$$\nabla^2 f(\mathbf{x}^*) \succeq 0$$

The **second-order sufficient condition** for a strict local minimum is:
$$\nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{and} \quad \nabla^2 f(\mathbf{x}^*) \succ 0$$

For convex $f$, the first-order condition is both necessary and sufficient for global optimality.

### Constrained Optimization: KKT Conditions

For problems with constraints, we use the **Lagrangian**:
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$$

where $\boldsymbol{\lambda} \in \mathbb{R}^m$ are **Lagrange multipliers** for inequality constraints and $\boldsymbol{\nu} \in \mathbb{R}^p$ are multipliers for equality constraints.

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary for optimality (under constraint qualifications):

1. **Stationarity**: $\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) = \mathbf{0}$

2. **Primal feasibility**: 
   - $g_i(\mathbf{x}^*) \leq 0$ for all $i$
   - $h_j(\mathbf{x}^*) = 0$ for all $j$

3. **Dual feasibility**: $\lambda_i^* \geq 0$ for all $i$

4. **Complementary slackness**: $\lambda_i^* g_i(\mathbf{x}^*) = 0$ for all $i$

For **convex optimization problems** (where $f$ and $g_i$ are convex, $h_j$ are affine), the KKT conditions are both necessary and sufficient for global optimality.

### Constraint Qualifications

KKT conditions are necessary under certain **constraint qualifications** that ensure regularity of the constraints. Common ones include:

**Linear Independence Constraint Qualification (LICQ)**: The gradients of active constraints are linearly independent at $\mathbf{x}^*$.

**Slater's condition** (for convex problems): There exists a strictly feasible point $\mathbf{x}$ with $g_i(\mathbf{x}) < 0$ for all $i$ and $h_j(\mathbf{x}) = 0$ for all $j$.

## Duality Theory

One of the most profound insights in optimization is **duality** — the idea that every optimization problem has an associated dual problem that provides bounds and complementary perspectives.

### The Lagrange Dual Function

Define the **Lagrange dual function**:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

Key property: $g$ is **concave** (even if the primal problem is non-convex) because it's the pointwise infimum of affine functions of $(\boldsymbol{\lambda}, \boldsymbol{\nu})$.

### Weak Duality

For any $\boldsymbol{\lambda} \geq \mathbf{0}$ and any $\boldsymbol{\nu}$:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f(\mathbf{x}^*)$$

where $\mathbf{x}^*$ is any primal optimal solution. This provides a **lower bound** on the optimal value.

### The Dual Problem

The **Lagrange dual problem** is:
$$\max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})$$
subject to: $\boldsymbol{\lambda} \geq \mathbf{0}$

The optimal value of the dual problem, $d^*$, satisfies:
$$d^* \leq p^*$$

where $p^*$ is the optimal value of the primal problem. The difference $p^* - d^*$ is called the **duality gap**.

### Strong Duality

**Strong duality** holds when $d^* = p^*$, i.e., the duality gap is zero.

For convex optimization problems satisfying Slater's condition, **strong duality holds**. This is one of the most important results in convex optimization theory.

### Dual Interpretations

The dual variables have important interpretations:
- $\lambda_i^*$ represents the **sensitivity** of the optimal value to changes in the $i$-th constraint
- $\nu_j^*$ represents the sensitivity to changes in the $j$-th equality constraint
- In economics, they're called **shadow prices** — the value of relaxing constraints

For the quadratic program:
$$\min_{\mathbf{x}} \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x} \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} = \mathbf{b}$$

The dual is:
$$\max_{\boldsymbol{\nu}} -\frac{1}{2}(\mathbf{Q}^{-1}(\mathbf{A}^T\boldsymbol{\nu} + \mathbf{c}))^T\mathbf{Q}(\mathbf{Q}^{-1}(\mathbf{A}^T\boldsymbol{\nu} + \mathbf{c})) + \mathbf{b}^T\boldsymbol{\nu}$$

## Classical Optimization Algorithms

### Gradient Descent

The most fundamental algorithm for unconstrained optimization:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$$

where $\alpha_k > 0$ is the **step size** or **learning rate**.

**Convergence for convex smooth functions**: If $f$ is convex and $L$-smooth ($\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$), then with fixed step size $\alpha = \frac{1}{L}$:
$$f(\mathbf{x}_k) - f(\mathbf{x}^*) \leq \frac{L\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2k}$$

This gives $O(1/k)$ convergence rate.

### Newton's Method

Newton's method uses second-order information:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

**Convergence**: Near a local minimum where $\nabla^2 f(\mathbf{x}^*) \succ 0$, Newton's method has **quadratic convergence**:
$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C\|\mathbf{x}_k - \mathbf{x}^*\|^2$$

This is much faster than gradient descent but requires computing and inverting the Hessian, which costs $O(n^3)$ per iteration.

### Conjugate Gradient Method

For quadratic problems $\min \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} - \mathbf{b}^T\mathbf{x}$, the conjugate gradient method:
- Requires no Hessian inversion
- Converges in at most $n$ iterations (exact arithmetic)
- Each iteration costs $O(n^2)$ (or less for sparse $\mathbf{Q}$)

The search directions $\mathbf{d}_k$ satisfy **$\mathbf{Q}$-conjugacy**:
$$\mathbf{d}_i^T \mathbf{Q} \mathbf{d}_j = 0 \quad \text{for } i \neq j$$

### Quasi-Newton Methods (BFGS)

BFGS approximates the Hessian inverse using only gradient information:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{H}_k \nabla f(\mathbf{x}_k)$$

where $\mathbf{H}_k$ approximates $[\nabla^2 f(\mathbf{x}_k)]^{-1}$ and is updated via:
$$\mathbf{H}_{k+1} = \mathbf{H}_k + \frac{\mathbf{s}_k\mathbf{s}_k^T}{\mathbf{s}_k^T\mathbf{y}_k} - \frac{\mathbf{H}_k\mathbf{y}_k\mathbf{y}_k^T\mathbf{H}_k}{\mathbf{y}_k^T\mathbf{H}_k\mathbf{y}_k}$$

with $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$ and $\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$.

BFGS achieves **superlinear convergence** while only requiring $O(n^2)$ per iteration.

## Constrained Optimization Algorithms

### Projected Gradient Descent

For constraints $\mathbf{x} \in \mathcal{C}$, project gradient steps onto the feasible set:
$$\mathbf{x}_{k+1} = \Pi_{\mathcal{C}}(\mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k))$$

where $\Pi_{\mathcal{C}}(\mathbf{y}) = \arg\min_{\mathbf{x} \in \mathcal{C}} \|\mathbf{x} - \mathbf{y}\|$ is the projection operator.

For simple sets (boxes, simplices, norm balls), projection can be computed efficiently.

### Penalty Methods

Convert constrained problem to unconstrained by penalizing constraint violations:
$$\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2}\sum_{i=1}^m \max(0, g_i(\mathbf{x}))^2 + \frac{\rho}{2}\sum_{j=1}^p h_j(\mathbf{x})^2$$

As penalty parameter $\rho \to \infty$, solutions converge to constrained optimum.

### Augmented Lagrangian Method

Combines Lagrange multipliers with penalty:
$$\mathcal{L}_{\rho}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \frac{\rho}{2}\sum_i \max(0, g_i(\mathbf{x}))^2 + \sum_j \nu_j h_j(\mathbf{x}) + \frac{\rho}{2}\sum_j h_j(\mathbf{x})^2$$

Alternately minimize over $\mathbf{x}$ and update multipliers:
$$\boldsymbol{\lambda}_{k+1} = \boldsymbol{\lambda}_k + \rho \max(0, g(\mathbf{x}_k))$$
$$\boldsymbol{\nu}_{k+1} = \boldsymbol{\nu}_k + \rho h(\mathbf{x}_k)$$

This converges with finite $\rho$, unlike pure penalty methods.

### Interior Point Methods

For inequality-constrained convex problems, interior point methods:
1. Stay strictly inside the feasible region
2. Follow the **central path** parameterized by $t > 0$:
   $$\mathbf{x}^*(t) = \arg\min_{\mathbf{x}} tf(\mathbf{x}) + \phi(\mathbf{x})$$
   where $\phi(\mathbf{x}) = -\sum_{i=1}^m \log(-g_i(\mathbf{x}))$ is the **logarithmic barrier**

3. Increase $t$ gradually while following the central path

Interior point methods achieve **polynomial-time** complexity for convex problems and are the basis for modern convex optimization solvers.

### Sequential Quadratic Programming (SQP)

For nonlinear constrained optimization, SQP:
1. Approximate problem locally by a quadratic program
2. Solve the QP to get search direction
3. Take a step in that direction with line search
4. Repeat

At iteration $k$, solve:
$$\min_{\mathbf{d}} \nabla f(\mathbf{x}_k)^T\mathbf{d} + \frac{1}{2}\mathbf{d}^T\nabla^2_{xx}\mathcal{L}(\mathbf{x}_k, \boldsymbol{\lambda}_k)\mathbf{d}$$
subject to:
$$\nabla g_i(\mathbf{x}_k)^T\mathbf{d} + g_i(\mathbf{x}_k) \leq 0$$
$$\nabla h_j(\mathbf{x}_k)^T\mathbf{d} + h_j(\mathbf{x}_k) = 0$$

SQP is the nonlinear analog of Newton's method and achieves superlinear convergence.

## Stochastic Optimization

Many modern applications involve optimization where the objective involves expectations or the gradient is expensive to compute exactly.

### Stochastic Gradient Descent (SGD)

Replace the true gradient with a stochastic estimate:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{g}_k$$

where $\mathbb{E}[\mathbf{g}_k | \mathbf{x}_k] = \nabla f(\mathbf{x}_k)$ but $\mathbf{g}_k$ is noisy.

For machine learning, this typically means computing gradients on mini-batches rather than the full dataset.

**Convergence for convex problems**: With step sizes $\alpha_k = \frac{1}{\sqrt{k}}$:
$$\mathbb{E}[f(\bar{\mathbf{x}}_k) - f(\mathbf{x}^*)] = O(1/\sqrt{k})$$

where $\bar{\mathbf{x}}_k = \frac{1}{k}\sum_{i=1}^k \mathbf{x}_i$ is the averaged iterate.

### Momentum Methods

Add momentum to smooth out noisy gradients:
$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k)$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{v}_{k+1}$$

where $\beta \in (0, 1)$ is the momentum parameter.

**Nesterov's Accelerated Gradient** uses a "look-ahead" gradient evaluation:
$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k - \alpha_k \beta \mathbf{v}_k)$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \mathbf{v}_{k+1}$$

This achieves $O(1/k^2)$ convergence for smooth convex functions.

### Adaptive Methods (Adam, RMSprop)

**Adam** (Adaptive Moment Estimation) combines momentum with adaptive learning rates:
$$\mathbf{m}_{k+1} = \beta_1 \mathbf{m}_k + (1 - \beta_1)\mathbf{g}_k$$
$$\mathbf{v}_{k+1} = \beta_2 \mathbf{v}_k + (1 - \beta_2)\mathbf{g}_k^2$$
$$\hat{\mathbf{m}}_{k+1} = \frac{\mathbf{m}_{k+1}}{1 - \beta_1^{k+1}}, \quad \hat{\mathbf{v}}_{k+1} = \frac{\mathbf{v}_{k+1}}{1 - \beta_2^{k+1}}$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \frac{\hat{\mathbf{m}}_{k+1}}{\sqrt{\hat{\mathbf{v}}_{k+1}} + \epsilon}$$

where $\beta_1, \beta_2 \in (0, 1)$ (typically 0.9, 0.999) and operations are element-wise.

## Non-Convex Optimization

Most deep learning problems are non-convex, raising fundamental challenges.

### Challenges of Non-Convexity

- Multiple local minima with different objective values
- Saddle points where $\nabla f(\mathbf{x}) = \mathbf{0}$ but $\nabla^2 f(\mathbf{x})$ has both positive and negative eigenvalues
- No guarantee that local search finds global optimum

### Why Deep Learning Works Despite Non-Convexity

Recent theory suggests several reasons:
1. **Over-parameterization**: High-dimensional problems may have no bad local minima
2. **Loss landscape geometry**: For neural networks, many local minima have similar objective values
3. **Implicit regularization**: SGD with proper initialization may favor good solutions
4. **Saddle points can be escaped**: Gradient descent with noise avoids saddle points

### Convergence to Stationary Points

For non-convex smooth functions, gradient descent with appropriate step sizes guarantees:
$$\min_{k=1,\ldots,K} \|\nabla f(\mathbf{x}_k)\|^2 \leq O(1/K)$$

This means we find an approximate stationary point, though it may not be a minimum.

## Applications to AI and Mobility Systems

### Training Neural Networks

The loss function for supervised learning:
$$\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{i=1}^N \ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i), y_i) + \lambda R(\boldsymbol{\theta})$$

where $\ell$ is the loss (e.g., cross-entropy), $f_{\boldsymbol{\theta}}$ is the neural network, and $R(\boldsymbol{\theta})$ is regularization.

This is solved via SGD or adaptive methods on mini-batches.

### Vehicle Routing Problems

The capacitated vehicle routing problem:
$$\min \sum_{(i,j) \in E} c_{ij} x_{ij}$$
subject to:
- Flow conservation constraints
- Capacity constraints: $\sum_{i \in S} q_i \leq Q$ for each vehicle
- Subtour elimination constraints

This is a mixed-integer program, solved via branch-and-bound or metaheuristics.

### Model Predictive Control

At each time step, solve:
$$\min_{\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}} \sum_{k=0}^{N-1} L(\mathbf{x}_k, \mathbf{u}_k) + V_f(\mathbf{x}_N)$$
subject to:
$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$$
$$\mathbf{x}_k \in \mathcal{X}, \quad \mathbf{u}_k \in \mathcal{U}$$

For linear systems with quadratic costs, this is a QP solved via interior point methods or active set methods.

### Reinforcement Learning

Policy gradient optimization:
$$\max_{\boldsymbol{\theta}} \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[\sum_{t=0}^{\infty} \gamma^t r(\mathbf{s}_t, \mathbf{a}_t)\right]$$

Solved via stochastic policy gradient methods:
$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \alpha \mathbb{E}[\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(\mathbf{a}_t|\mathbf{s}_t) A^{\pi}(\mathbf{s}_t, \mathbf{a}_t)]$

where $A^{\pi}$ is the advantage function.

### Traffic Network Optimization

System-optimal traffic assignment:
$\min \sum_{a \in A} \int_0^{f_a} t_a(w) dw$
subject to:
$\sum_{p \in P_k} h_p^k = d_k \quad \forall k \in K$ (demand satisfaction)
$f_a = \sum_{k \in K} \sum_{p \in P_k} \delta_{ap}^k h_p^k \quad \forall a \in A$ (flow conservation)
$h_p^k \geq 0 \quad \forall p, k$

where $t_a(f_a)$ is the travel time on link $a$ as a function of flow, $h_p^k$ is flow on path $p$ for origin-destination pair $k$, and $\delta_{ap}^k = 1$ if link $a$ is on path $p$ for OD pair $k$.

This is a convex optimization problem when $t_a(\cdot)$ are increasing convex functions.

## Advanced Topics: Distributed and Parallel Optimization

Modern large-scale systems often require distributed optimization where computation is spread across multiple agents or processors.

### Alternating Direction Method of Multipliers (ADMM)

For problems with separable structure:
$\min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z})$
subject to: $\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$

ADMM alternates between:
$\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z}_k - \mathbf{c} + \mathbf{u}_k\|^2$
$\mathbf{z}_{k+1} = \arg\min_{\mathbf{z}} g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x}_{k+1} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{u}_k\|^2$
$\mathbf{u}_{k+1} = \mathbf{u}_k + \mathbf{A}\mathbf{x}_{k+1} + \mathbf{B}\mathbf{z}_{k+1} - \mathbf{c}$

ADMM is particularly useful for:
- Distributed optimization across multiple agents
- Large-scale machine learning with regularization
- Consensus problems in multi-agent systems

### Consensus Optimization

For $N$ agents each with local objective $f_i(\mathbf{x})$, the consensus problem is:
$\min_{\mathbf{x}} \sum_{i=1}^N f_i(\mathbf{x})$

Each agent maintains local copy $\mathbf{x}_i$ and iterates:
$\mathbf{x}_i^{k+1} = \mathbf{x}_i^k - \alpha \nabla f_i(\mathbf{x}_i^k) + \beta \sum_{j \in \mathcal{N}_i} (\mathbf{x}_j^k - \mathbf{x}_i^k)$

The second term enforces consensus through local communication. Under appropriate conditions on the communication graph, all $\mathbf{x}_i^k$ converge to the same optimal value.

## Derivative-Free and Black-Box Optimization

When gradients are unavailable or unreliable, we need alternative approaches.

### Nelder-Mead Simplex Method

Maintains a simplex (set of $n+1$ points in $\mathbb{R}^n$) and iteratively:
1. **Reflect** the worst point through the centroid
2. **Expand** if reflection is good
3. **Contract** if reflection is poor
4. **Shrink** all points toward the best if contraction fails

No gradient required, but convergence can be slow and is not guaranteed for non-convex problems.

### Genetic Algorithms and Evolutionary Methods

Maintain a population of candidate solutions and:
1. **Selection**: Choose better solutions with higher probability
2. **Crossover**: Combine pairs to create offspring
3. **Mutation**: Random perturbations for exploration
4. **Replacement**: Update population

Useful for combinatorial and highly non-convex problems (e.g., neural architecture search).

### Bayesian Optimization

Model the objective as a Gaussian process and iteratively:
1. Update posterior belief based on observations
2. Choose next point to evaluate using an **acquisition function** (e.g., expected improvement)
3. Observe objective value and update model

Particularly effective for expensive black-box functions (e.g., hyperparameter tuning).

## Online and Bandit Optimization

In dynamic environments, we must optimize while learning about the system.

### Online Convex Optimization

At each round $t = 1, 2, \ldots$:
1. Choose $\mathbf{x}_t$ from feasible set $\mathcal{K}$
2. Observe convex loss function $f_t$
3. Incur loss $f_t(\mathbf{x}_t)$

Goal: Minimize **regret**:
$R_T = \sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}^* \in \mathcal{K}} \sum_{t=1}^T f_t(\mathbf{x}^*)$

**Online Gradient Descent** achieves $R_T = O(\sqrt{T})$ regret:
$\mathbf{x}_{t+1} = \Pi_{\mathcal{K}}(\mathbf{x}_t - \eta_t \nabla f_t(\mathbf{x}_t))$

This is the online analog of batch optimization and connects to reinforcement learning in non-stationary environments.

### Multi-Armed Bandits

Choose among $K$ actions (arms) to maximize cumulative reward. At each round:
1. Choose arm $a_t$
2. Observe reward $r_t(a_t)$
3. Update estimates

**Upper Confidence Bound (UCB)** algorithm:
$a_t = \arg\max_a \left[\hat{\mu}_a(t) + \sqrt{\frac{2\log t}{n_a(t)}}\right]$

where $\hat{\mu}_a(t)$ is the empirical mean reward of arm $a$ and $n_a(t)$ is the number of times it's been played.

UCB achieves $O(\log T)$ regret, which is optimal for stochastic bandits.

## Structured Optimization Problems

### Linear Programming (LP)

$\min_{\mathbf{x}} \mathbf{c}^T\mathbf{x}$
subject to: $\mathbf{A}\mathbf{x} = \mathbf{b}$, $\mathbf{x} \geq \mathbf{0}$

**Simplex method**: Navigate vertices of the feasible polytope
**Interior point methods**: Follow central path through the interior

LPs are fundamental for operations research, logistics, and resource allocation.

### Quadratic Programming (QP)

$\min_{\mathbf{x}} \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x}$
subject to: $\mathbf{A}\mathbf{x} = \mathbf{b}$, $\mathbf{G}\mathbf{x} \leq \mathbf{h}$

If $\mathbf{Q} \succeq 0$, this is convex and solvable efficiently. Used extensively in MPC and portfolio optimization.

### Semidefinite Programming (SDP)

$\min_{\mathbf{X}} \langle \mathbf{C}, \mathbf{X} \rangle$
subject to: $\langle \mathbf{A}_i, \mathbf{X} \rangle = b_i$, $\mathbf{X} \succeq 0$

where $\mathbf{X} \in \mathbb{S}^n$ is a symmetric matrix and $\mathbf{X} \succeq 0$ means positive semidefinite.

SDPs generalize LPs and QPs and arise in:
- Robust control synthesis
- Combinatorial optimization relaxations
- Quantum information theory
- Machine learning (kernel methods)

### Mixed-Integer Programming (MIP)

$\min_{\mathbf{x}, \mathbf{y}} \mathbf{c}^T\mathbf{x} + \mathbf{d}^T\mathbf{y}$
subject to: $\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{y} \leq \mathbf{b}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{y} \in \mathbb{Z}^m$

Some variables are continuous, others integer. NP-hard in general, but practical instances with thousands of variables can be solved via:
- **Branch and bound**: Systematically partition and prune search space
- **Cutting planes**: Add constraints that tighten LP relaxation
- **Branch and cut**: Combine both approaches

Critical for scheduling, routing, facility location, and discrete decision-making.

## The Optimization Hierarchy

Different problem classes form a hierarchy of tractability:

$\text{LP} \subset \text{QP} \subset \text{SOCP} \subset \text{SDP} \subset \text{Convex} \subset \text{Non-convex}$

where SOCP is **Second-Order Cone Programming**. Each inclusion represents increasing generality but decreasing tractability.

**Cone programming** unifies these:
$\min_{\mathbf{x}} \mathbf{c}^T\mathbf{x}$
subject to: $\mathbf{A}\mathbf{x} + \mathbf{b} \in \mathcal{K}$

where $\mathcal{K}$ is a convex cone:
- $\mathcal{K} = \mathbb{R}_+^n$: Linear programming
- $\mathcal{K} = \mathbb{Q}^n$ (second-order cone): SOCP
- $\mathcal{K} = \mathbb{S}_+^n$ (PSD cone): SDP

## Connecting Optimization to Control and Learning

The mathematical frameworks we've explored reveal deep connections:

### Optimal Control as Optimization

Continuous-time optimal control:
$\min_{\mathbf{u}(\cdot)} \int_0^T L(\mathbf{x}(t), \mathbf{u}(t)) dt$
subject to: $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$

This is an infinite-dimensional optimization problem over function spaces. The necessary conditions (Pontryagin's Maximum Principle, HJB equation) are the optimization optimality conditions specialized to this setting.

### Machine Learning as Optimization

Supervised learning:
$\min_{\boldsymbol{\theta}} \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[\ell(f_{\boldsymbol{\theta}}(\mathbf{x}), y)]$

This is stochastic optimization where we approximate the expectation with samples. The entire field of deep learning is fundamentally about solving large-scale non-convex stochastic optimization problems.

### Reinforcement Learning as Online Optimization

RL can be viewed as online convex optimization in policy space:
$\max_{\pi} \mathbb{E}_{\tau \sim \pi}[R(\tau)]$

The agent doesn't know the objective initially and must learn through interaction, similar to online optimization where loss functions are revealed sequentially.

### Multi-Agent Systems as Game-Theoretic Optimization

Each agent $i$ solves:
$\min_{\mathbf{u}_i} J_i(\mathbf{u}_1, \ldots, \mathbf{u}_N)$

Nash equilibria correspond to solutions where each agent's optimization problem is solved given others' actions. This connects optimization to game theory and mechanism design.

## Practical Considerations and Software

Modern optimization relies heavily on high-quality software:

**Convex optimization**:
- CVXPY (Python): Modeling language for convex problems
- CVX (MATLAB): Domain-specific language with automatic problem recognition
- Solvers: MOSEK, Gurobi, CPLEX for commercial applications

**Non-convex optimization** (deep learning):
- PyTorch, TensorFlow: Automatic differentiation and GPU acceleration
- JAX: Composable transformations for high-performance computing

**Mixed-integer programming**:
- Gurobi, CPLEX: Commercial solvers with sophisticated branch-and-cut
- SCIP, CBC: Open-source alternatives

**Derivative-free**:
- SciPy: Nelder-Mead, Powell's method
- Optuna, Hyperopt: Bayesian optimization for hyperparameter tuning

## Looking Ahead: Game Theory and Multi-Agent Systems

We've now established three foundational pillars:
1. **Dynamic systems**: How systems evolve over time
2. **Control theory**: How to shape system behavior
3. **Optimization theory**: The mathematical language of decision-making

In our next post, we'll explore what happens when multiple decision-makers interact — the realm of **game theory and multi-agent systems**. We'll see how optimization generalizes to competitive and cooperative scenarios, how Nash equilibria emerge as the solution concept, and how these ideas apply to coordinating autonomous vehicles, managing traffic networks, and designing multi-agent AI systems.

But before we move forward, it's worth appreciating the profound unification that optimization theory provides. Whether we're training a neural network, controlling a robot, planning a route, or allocating resources, we're fundamentally solving optimization problems. The mathematical tools we've developed — from convexity and duality to gradient methods and constraint handling — are the common language that connects seemingly disparate fields.

This universality is why optimization theory is so powerful. It doesn't just solve specific problems; it provides a way of thinking about decision-making mathematically. And as we'll see in future posts, when we extend these ideas to multi-agent settings, to learning under uncertainty, and to hierarchical decision-making, optimization remains at the core — connecting classical operations research to modern artificial intelligence, and providing the mathematical foundation for the intelligent systems of tomorrow.

---

**References**

**Boyd, S., & Vandenberghe, L.** (2004). *Convex optimization*. Cambridge University Press. [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/)

**Nocedal, J., & Wright, S. J.** (2006). *Numerical optimization* (2nd ed.). Springer.

**Bertsekas, D. P.** (2016). *Nonlinear programming* (3rd ed.). Athena Scientific.

**Nesterov, Y.** (2018). *Lectures on convex optimization* (2nd ed.). Springer.

**Bubeck, S.** (2015). *Convex optimization: Algorithms and complexity*. Foundations and Trends in Machine Learning, 8(3-4), 231–357.

**Kingma, D. P., & Ba, J.** (2015). Adam: A method for stochastic optimization. *Proceedings of ICLR 2015*.

**Shalev-Shwartz, S.** (2012). *Online learning and online convex optimization*. Foundations and Trends in Machine Learning, 4(2), 107–194.

**Wolsey, L. A.** (2020). *Integer programming* (2nd ed.). Wiley.