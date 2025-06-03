# Stochastic Nested Bandits

This repository provides a stochastic environment for simulating interactions in the context of **Nested Bandits** — a variant of the multi-armed bandit problem with a hierarchical or nested structure.

## Overview

The project aims to model and simulate decision-making scenarios where actions are nested and feedback is obtained in a stochastic manner. It serves as a testbed for developing and evaluating algorithms designed to solve the **Stochastic Nested Bandit Problem**, which arises in various real-world applications such as recommendation systems, online advertising, and hierarchical decision processes.

![Nested Environment](https://github.com/AugustinCablant/CRITEO-INTERNSHIP/blob/main/tree_mu_baselines.png)


## Features

- Simulation of nested decision structures with stochastic rewards  
- Customizable environments for experimentation  
- Easily extensible for testing new algorithms  

## Implementation of Baselines

| Algorithm         | Setting                | Reference                                                                 |
|------------------|------------------------|---------------------------------------------------------------------------|
| UCB              | Stochastic             | [Lattimore & Szepesvári (2020)](https://tor-lattimore.com/downloads/book/book.pdf) |
| Thompson Sampling| Stochastic             | [Russo et al. (2018)](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf) |
| KL-UCB           | Stochastic             | [Garivier & Cappé (2011)](https://arxiv.org/pdf/1102.2490)               |
| Exp3             | Adversarial            | [Cesa-Bianchi & Lugosi (2006)](https://sequential-learning.github.io/docs/online_learning_lecture_notes.pdf#page=5) |
| Exp3.P           | Adversarial            | [Bubeck & Cesa-Bianchi (2012)](http://sbubeck.com/SurveyBCB12.pdf)       |
| Exp3++           | Adversarial            | [Seldin & Lugosi (2017)](https://proceedings.mlr.press/v65/seldin17a/seldin17a.pdf) |
| SAO              | Best of Both Worlds    | [Bubeck et al. (2012)](http://proceedings.mlr.press/v23/bubeck12b/bubeck12b.pdf) |
| Tsallis-INF      | Best of Both Worlds    | [Zimmert & Seldin (2019)](https://arxiv.org/pdf/1807.07623)              |
| NEW              | Adversarial            | [Zimmert & Lattimore (2022)](https://arxiv.org/pdf/2206.09348)           |
| NUCB             | Stochastic (Nested)    |                                                                          |



## Use Cases

- Testing algorithms for stochastic nested bandits
- Benchmarking bandit strategies in structured environments
- Research and prototyping for hierarchical decision systems

## License

This project is open-source and available under the MIT License.

