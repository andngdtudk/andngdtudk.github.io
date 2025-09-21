<!-- image: https://andngdtudk.github.io/images/computer.jpg -->

# What Is a System? — Definitions, Characteristics, and Classifications

_Copenhagen_, 21st September 2025

In my previous post, I traced humanity's journey from ancient irrigation networks to modern Physical AI systems. Throughout this progression, I kept using the word "system" — but what exactly do we mean by it?

Consider these scenarios: A flock of birds suddenly changes direction mid-flight. A traffic jam forms on an empty highway for no apparent reason. A supply chain disruption in one region causes shortages thousands of miles away. An autonomous vehicle fleet learns to coordinate without explicit communication. What connects these seemingly disparate phenomena? They are all examples of systems in action — collections of interacting parts that produce behaviors and outcomes that none of the individual parts could achieve alone.

<p align="center" style="margin-top: 20px; margin-bottom: 10px;">
    <img src="images/bird_flock-01.png" alt="Flock of birds in coordinated motion" style="display: block; margin: auto; width: 75%;">
</p>

<figcaption style="text-align: center;">
    Figure 1: A murmuration of starlings forming complex, fluid patterns in the sky. 
    <br>
    <span style="font-size: 0.9em;">
        This emergent behavior arises from simple local rules followed by each bird — aligning with neighbors, avoiding collisions, and staying close — producing a collective motion that appears highly coordinated at the system level.
    </span>
    <br>
    <span style="font-size: 0.8em; font-style: italic; margin-bottom: 20px;">
        Source: <a href="https://www.howitworksdaily.com/why-do-birds-flock-together/">How it works</a>
    </span>
</figcaption>

Understanding what makes something a "system" is crucial for anyone working with complex, dynamic environments. Whether you're designing an AI agent, optimizing a logistics network, or trying to predict emergent behaviors in multi-agent scenarios, the principles of systems thinking provide the foundational lens through which to approach these challenges.

## Defining a System: More Than the Sum of Its Parts

At its most basic level, a system is a collection of interrelated elements that work together toward a common purpose or function. But this definition, while accurate, doesn't capture what makes systems so fascinating and challenging to work with.

Systems theorist Donella Meadows offered a more nuanced definition: "A system is an interconnected set of elements that is coherently organized in a way that achieves something" (Meadows, 2008). This definition highlights three essential components: **elements**, the individual parts or components; **interconnections**, the relationships and interactions between elements; and **purpose or function**, the overall behavior or goal that emerges from these interactions.

But here's what makes systems truly interesting: the behavior of the whole system cannot be predicted simply by understanding the individual parts. A traffic network isn't just a collection of roads — it's the flow patterns, congestion dynamics, and routing behaviors that emerge from the interaction of vehicles, infrastructure, and control mechanisms. An autonomous vehicle fleet isn't just individual cars with AI — it's the coordination patterns, resource allocation strategies, and collective intelligence that arise from their interactions.

This property — where the whole exhibits characteristics that are not present in any individual part — is called **emergence**. And it's why systems thinking is so essential for understanding complex, intelligent systems.

## Core Characteristics of Systems

All systems, from biological ecosystems to transportation networks to multi-agent AI systems, share certain fundamental characteristics:

**Structure and organization**: Systems have an internal structure — a pattern of relationships and hierarchies that determine how information, energy, or resources flow through the system. In a logistics network, this might be the hub-and-spoke topology of distribution centers. In a reinforcement learning system, it could be the architecture of neural networks and the flow of experience data.

**Behavior and function**: Systems exist to perform some function or exhibit specific behaviors. This might be explicit (like a traffic control system designed to minimize travel time) or implicit (like the emergent coordination behaviors in a swarm of autonomous drones). Understanding a system's purpose is crucial for predicting and controlling its behavior.

**Boundaries and environment**: Every system has boundaries that separate it from its environment. These boundaries define what is "inside" the system versus what is external. For an autonomous vehicle, the boundary might include its sensors, processing units, and actuators, while the environment includes other vehicles, road infrastructure, and weather conditions. Importantly, systems are rarely completely closed — they exchange information, energy, or materials with their environment.

**Feedback loops**: Systems contain feedback mechanisms that allow them to respond to changes and maintain stability or adapt to new conditions. Negative feedback helps maintain equilibrium (like a thermostat maintaining temperature), while positive feedback can amplify changes (like traffic congestion that begets more congestion). In AI systems, feedback often comes through reward signals or performance metrics that guide learning and adaptation.

**Dynamic behavior over time**: Systems are not static — they evolve, adapt, and change over time. This temporal dimension is what makes systems both powerful and challenging to control. A traffic network behaves differently during rush hour versus late at night. An AI system's behavior changes as it learns from new experiences.

## Classifications of Systems

Not all systems are created equal, and understanding different types of systems helps us choose appropriate modeling and control approaches.

**Simple vs. Complex Systems**: Simple systems have few components with straightforward interactions — a pendulum or a basic feedback controller is an example. Complex systems have many components with nonlinear interactions that can produce surprising behaviors, like urban transportation networks or large-scale AI systems.

**Static vs. Dynamic Systems**: Static systems do not change over time, while dynamic systems evolve. Most interesting real-world systems are dynamic — an autonomous vehicle's perception of its environment is constantly updating.

**Deterministic vs. Stochastic Systems**: Deterministic systems produce predictable outputs given specific inputs, while stochastic systems involve randomness or uncertainty. Real-world systems almost always have stochastic components: sensor noise, unpredictable human behavior, or environmental variation.

**Open vs. Closed Systems**: Closed systems do not exchange anything with their environment, while open systems do — and most practical systems are open, receiving inputs and sending outputs.

<p align="center" style="margin-top: 20px; margin-bottom: 10px;">
    <img src="images/open_close_system.png" alt="Open-loop and closed-loop control systems" style="display: block; margin: auto; width: 70%;">
</p>

<figcaption style="text-align: center; margin-bottom: 20px;">
    Figure 2: Open-loop (top) and closed-loop (bottom) control systems. 
    <span style="font-size: 0.9em;">
        In an open-loop system, the controller issues a control action based solely on the input, without considering the actual output. In contrast, a closed-loop (feedback) system continuously measures the output and uses feedback to adjust the control action, improving stability, accuracy, and robustness in the presence of disturbances.
    </span>
</figcaption>

**Centralized vs. Distributed Systems**: Centralized systems have a single point of control, while distributed systems spread decision-making across multiple nodes or agents. Modern AI and mobility applications increasingly rely on distributed approaches — think of a smart city where traffic optimization is handled locally at intersections that coordinate with one another.

**Single-Agent vs. Multi-Agent Systems**: Another important dimension of classification considers the number of decision-makers. A single-agent system has one entity making decisions to achieve its objectives, such as a thermostat regulating temperature or a reinforcement learning agent learning to play a video game. A multi-agent system, by contrast, consists of multiple decision-makers, each potentially with its own goals and information. Examples include fleets of autonomous vehicles, power grids with many independent participants, or social and economic systems. Multi-agent systems are inherently more complex because the environment becomes non-stationary — as each agent adapts, it changes the conditions faced by the others — often requiring coordination, negotiation, or game-theoretic reasoning.

## Why These Distinctions Matter for AI and Mobility

Understanding these classifications is not just an academic exercise. It directly informs how we model, control, and optimize real-world systems. A complex, stochastic, dynamic system requires different mathematical tools than a simple, deterministic one. Centralized strategies that work in closed systems can fail in open, distributed environments. The behavior of complex systems can be fundamentally unpredictable in detail, which is why robust optimization, uncertainty quantification, and decentralized learning strategies have become central to modern research.

Recognizing whether a behavior is emergent or designed determines whether we engineer better components or focus on shaping the rules of interaction. In mobility, this distinction could mean the difference between improving the perception system of an autonomous vehicle (a component-level intervention) and redesigning how multiple vehicles coordinate at an intersection (a system-level intervention).

## The Systems View of Intelligence

This systems perspective leads to a profound insight: intelligence itself can be understood as a system property. Individual neural network weights are not "intelligent," but their organized interactions produce intelligent behavior. Similarly, a single vehicle following rules is not remarkable, but a fleet of vehicles coordinating, learning, and adapting to new conditions exhibits collective intelligence that emerges from their interactions.

## Looking Ahead

In our next post, we'll explore how these systems concepts apply specifically to **dynamic systems** — systems that evolve over time. We'll look at state spaces, system dynamics, and the mathematical frameworks that allow us to predict and control complex, time-varying behaviors.

Before we get there, it is worth noting that recognizing something as a system fundamentally changes how we approach it. Instead of focusing on components in isolation, we look for patterns of interaction. Instead of trying to control every detail, we search for leverage points where small changes can lead to big effects. And instead of insisting on perfect prediction, we embrace adaptability and resilience.

This shift — from parts to wholes, from control to influence, from prediction to adaptation — is at the heart of systems thinking. It is also essential for navigating the complex, uncertain, multi-agent world of modern AI and mobility systems.

## References

Meadows, D. H. (2008). *Thinking in systems: A primer*. Chelsea Green Publishing.  
Mobus, G. E., & Kalton, M. C. (2015). *Principles of systems science*. Springer.  
von Bertalanffy, L. (1968). *General system theory: Foundations, development, applications*. George Braziller.  
Checkland, P. (1999). *Systems thinking, systems practice*. John Wiley & Sons.








