---
layout: post
title:  "Hypergradient Optimization"
date: 2019-06-27
author: nandan_prince
excerpt: "A brief introduction about the Hypergradient optimization for Hyperparameter optimization"
tags: ["Automl","Hyperparameter optimization", "Hypergradient"]
comments: True
mathjax: True
---
<h2>Introduction</h2>

The increasing complexity of machine learning algorithms has driven a large amount of research in the area of hyperparameter optimization(HO). The core idea is to use a validation set to construct a response of the hyperparameters and explore the hyperparameter space to seek the optimum. Early approaches based on grid search quickly become impractical as the number of hyperparameter increases and are even outperformed by random search. Given high computation cost of evaluating the response function, Bayesian optimization approaches provide a natural framework and has been extensively studied. 
Here we'll discuss an alternative approach where gradient-based algorithms are used to optimize the performance on a validation set with respect to hyperparameters. The validation error should be evaluated at a minimizer of the training objective.

To understand any of the approach stated above, let's first understand what the actual problem of optimization is, let's try to reach at a mathematical expression of the problem. 
We focus on training procedures based on the optimization of an objective function $L$ with respect to $w$ (e.g. the regularized average training loss for the neural nets with weights $w$). We see the training procedure by stochastic gradient descent (or one of its variants like momentum, RMSProp, Adam, etc.) as a dynamical system with a state $s_t$ $\in$ $R^d$ that collects weights and possible accessory variables. The dynamics are defined by the system of equations: 

$$ s_t = \Phi_t(s_{t-1}, \lambda),    t = 1,...,T $$

where $T$ is the number of itterations, $s_0$ contains initial weights and initial accessory variables and $t \in {1,...,T}$ is the $t$-th step of the optimization algorithm,i.e. on mini-batch $t$. Finally, $\lambda$ $\in$ $R^m$ is the vector of hyperparameters that we wish to tune.
 
let's take a simple example training a neural network by gradient descent with momentum(GDM), in which case $s_t = (v_t, w_t)$ and, 


$$v_t = \mu v_{t-1} + \nabla J_t(w_{t-1})$$

$$w_t = w_{t-1} - \eta (\mu v_{t-1} - \nabla J_t(w_{t-1}))$$

where $J_t$ is the objective function associated with the $t$-th mini-batch, $\mu$ is the rate and $\eta$ is the momentum. In this example, $\lambda = (\mu, \eta)$.
Here $s_1,...,s_T$ implicitly depend on the vector of hyperparameter $\lambda$. Our goal is to optimize this hyperparameter according to certain error function $E$ evaluated at last iterate $s_T$. We wish to solve the problem
$$ \min{\lambda \in \Lambda} f(\lambda)$$
where the set $\Lambda$ incorporates constraints on the hyperparameters, and the response function $f$ is defined at $\lambda$ as 
$ f(\lambda) = \textit{E}(s_T(\lambda))\$.
Now to minimize $f$($\lambda$) we first have to compute the gradient of $f(\lambda)$(or hypergradient) with respect to $\lambda$, which can be computed by chain rule,

$$\nabla f(\lambda) = \nabla E(s_T) \frac{ds_T}{d\lambda}$$

where $\frac{ds_T}{d \lambda}$ is formed by total derivative of the component of $s_T$ with respect to the component of $\lambda$. As $s_t = (s_{t-1}, \lambda)$, the operators $\Phi_t$ depends on the hyperparameter $\lambda$ both directly by its expression and indirectly through the state $s_{t-1}$. Using again the chain rule we have, for every $t \in {1,...,T}$,

$$ \frac{ds_t}{d\lambda} = \frac{\partial \Phi_t(s_{t-1})}{\partial s_{t-1}} \frac{ds_{t-1}}{d \lambda} + \frac{ds_t}{d \lambda} $$

Defining,

$$
    Z_t=\frac{ds_t}{d\lambda},  A_t = 
    \frac{\partial\Phi_t(s_{t-1})} {\partial s_{t-1}}, 
    \textit{B_t} = \frac{ds_t}{d \lambda}
$$
the equation can be written as,

$$
    \nabla f(\lambda) = \nabla E(s_T) Z_t = \nabla E(s_T) (A_t Z_{t-1} + B_t)  
$$

This can be calculated by recursively calculating $Z_t$, which lead to the final expression:

$$
     \nabla f(\lambda) = \nabla\textit{E(s_T)}  \sum_{t=1}^{T} (\prod_{s=t+1}^{T} A_s)B_t
$$

Now as we have calculated the hypergradient we can update our hyperparameter using this after every epoch of training. 
