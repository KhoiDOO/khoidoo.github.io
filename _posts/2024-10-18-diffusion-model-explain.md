---
layout: post
title: Diffusion Models - A Cleaned Mathematical Explaination
date: 2024-10-18 11:12:00-0400
description: A Cleaned Mathematical Explaination
tags: diffusion
categories: generative-ai
related_posts: true
---

# Introduction
Diffusion models have emerged as a powerful class of probabilistic generative models, demonstrating state-of-the-art performance in various machine learning tasks, particularly image synthesis. These models are grounded in the principles of stochastic processes, leveraging the concepts of forward and reverse diffusion to model data distributions. Diffusion models operate by progressively adding noise to data (forward diffusion) until it becomes a simple distribution, typically Gaussian. The reverse diffusion process then learns to denoise this noisy data back to its original form. This technical report provides an in-depth analysis of diffusion models, covering their theoretical underpinnings, algorithmic implementations, and practical applications. It is inspired by Lilian Weng's comprehensive blog post "What are Diffusion Models?", which offers a detailed and accessible explanation.

# Denoising Diffusion Probabilistic Model (DDPM)
Diffusion models\citep{diffusion} are latent variable models which have the form as follows:

$$
\begin{align}
    p_\theta(x_0) = \int{p_\theta(x_{0:T})dx_{1:T}},
\end{align}
$$

which is also the probability the generative model assigns to the data, $$x_1, \dots, x_T$$ are latent vectors of the same dimensionality as the original data $$x_0 \sim q(x_0)$$, as the prior distribution. Otherwise, $$\theta$$ is the model parameter, $$T$$ is the total number of timesteps in the Markov chain process. 
<!-- \begin{figure}[!ht]
    \centering
    \includegraphics[width=\textwidth]{../assets/img/posts/2024-10-18-diffusion-model-explain/ddpm.pdf}
    \caption{Markov chain simulation. DDPM contains two processes: reverse and forward, which are $p_\theta(x_{t-1}\mid x_t)$ and $q(x_t\mid x_{t-1})$. DDPM seeks to approximate $q(x_{t-1}\mid x_t)$ to reverse back to $x_{t-1}$ from $x_t$. The original version from \citep{ddpm}.}
    \label{fig:ddpm}
\end{figure} -->
[blal](assets/img/posts/2024-10-18-diffusion-model-explain)

The idea behind DDPM is to gradually remove noise from a data sample (forward process), making it clearer step by step (reverse process) (refer to Figure \ref{fig:ddpm}). The underlying issue is how to approximate the transition from noise back to the original data point. This technical report will undercover mathematically both forward and reverse processes, especially in the reverse process the loss function construction, conditional forward transition estimation, and its parameterized version.

# Forward Processs
Diffusion models\citep{ddpm} use of a specific approximate posterior,  $$q(x_{1:T} \mid x_0)$$, known as the* forward process} or *diffusion process}. This process is predetermined as a Markov chain that incrementally introduces Gaussian noise to the sample in $$T$$ steps, following a predefined variance schedule $$ \beta_1, \ldots, \beta_T $$ as $$ \beta_t \in (0, 1) $$ for $$ t = 1, \ldots, T $$.

$$
\begin{align}
    q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_{t} \mid x_{t-1}),
\end{align}
$$

$$
\begin{align}
    q(x_{t} \mid x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\textbf{I})
\end{align}
$$

The intuition behind $$\sqrt{1-\beta_t}$$ is that authors\citep{ddpm} would like to keep the latent variance in all timestep constant, specifically to 1. Considering an unknown variance across timesteps $$\nu_t$$, hence the density of $$x_t$$ conditioned on $$x_{t-1}$$ is as follows:

$$
\begin{align}
    q(x_{t} \mid x_{t-1}) = \mathcal{N}(x_t;\nu_t x_{t-1}, \beta_t\textbf{I})
\end{align}
$$

The variance of $$x_t$$ is as follows:

$$
\begin{align}\label{eq:var_x_t}
    Var[x_t] &= Var\left[\nu_t x_{t-1} + \sqrt{\beta_t}\epsilon_t\right] \\
    &= \nu_t^2Var\left[x_{t-1}\right] + \beta_tVar\left[\epsilon_t\right] + \nu_t^2\beta_tCov\left[x_{t-1}, \epsilon_t\right]
\end{align}
$$

Since $$p_{t-1}$$ and $$p(\epsilon_t)$$ are independent, thus $$p(x_{t-1}, \epsilon_t) = p(x_{t-1})p(\epsilon_t)$$ and $$Cov[x_{t-1}, \epsilon_t] = 0$$. Substituting into Equation \eqref{eq:var_x_t}, we have:

$$
\begin{align}
    Var[x_t] &= Var\left[\nu_t x_{t-1} + \sqrt{\beta_t}\epsilon_t\right] \\
    &= \nu_t^2 Var\left[x_{t-1}\right] + \beta_t Var\left[\epsilon_t\right] \\
    &= \nu_t^2 Var\left[x_{t-1}\right] + \beta_t \quad\textrm{since $\epsilon\sim\mathcal{N}(0, \textbf{I})$}
\end{align}
$$

Given that, we would like the variance of latent across timesteps to be $$1$$, so we set $$Var[x_t] = Var[x_{t-1}] = 1$$, we have

$$
\begin{align}
    1 &= \nu_t^2 + \beta_t\\
    \Leftrightarrow \nu_t &= \sqrt{1 - \beta_t} 
\end{align}
$$

As the time step $$T$$ increases, the distinguishable features of the initial data sample $$x_0$$ gradually diminish (refer to Figure \ref{fig:ddpm}). As $$T$$ approaches infinity, $$x_T$$ converges to an isotropic Gaussian distribution. A beneficial aspect of this process is that it allows for sampling $$x_t$$ at any chosen time step $$T$$ in a closed form by utilizing the reparameterization trick \citep{vae}. 

$$
\begin{align}
    x_t &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}; \quad \textrm{where} \quad \epsilon_{t-1}, \epsilon_{t-2},\ldots~ \mathcal{N}(0, \textbf{I})\\
    &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha}\epsilon_{t-1}; \quad \textrm{where} \quad \alpha_t = 1 - \beta_t \quad and \quad \bar{\alpha} =\prod_{t=1}^{T}\alpha_i\\
    &= \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \sqrt{1- \alpha_t \alpha_{t-1}}\bar{\epsilon}_{t-2}; \quad\textrm{where} \quad\bar{\epsilon}_{t-2} \quad \textrm{ is merged two Gaussians(*)}\\
    &= \ldots\notag\\
    &= \sqrt{\bar\alpha_t}x_0 + \sqrt{1- \bar\alpha_t}\epsilon
\end{align}
$$

As a result,
$$
\begin{align}
    q(x_{t} \mid x_{0}) = \mathcal{N}(x_t;\sqrt{\bar\alpha_t}x_0, (1- \bar\alpha_t)\textbf{I})
\end{align}
$$

Typically, a larger update step can be taken when the sample becomes noisier, so $$\beta_1 < \beta_2<\ldots< \beta_T$$ and therefore $$\bar\alpha_1> \bar\alpha_2>\ldots>\bar\alpha_T $$.

## Reverse Process
The *reverse process} (refers to Equation \eqref{eq:rev}) is the joint distribution $$p_\theta(x_{0:T})$$, defined as a Markov chain with *learned Gaussian transitiion} starting at $$p(x_T) = \mathcal{N}(x_T, 0, \textbf{I})$$. By conducting the reverse process, the model will be able to recreate the true sample from a Gaussian noise input, $$x_T \sim \mathcal{N}(0, \textbf{I})$$. 

$$
\begin{align}\label{eq:rev}
    p_\theta(x_{0:T}) = p(x_T)\prod p_\theta(x_{t-1}\mid x_t)
\end{align}
$$

The transition probability in the reverse process is then approximated by model $$\theta$$ as follows:

$$
\begin{align}
    p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1}; \boldsymbol{\mu}_\theta(x_t, t), \boldsymbol{\Sigma}_\theta(x_t, t))
\end{align}
$$

where the transition probability distribution is expected to be Gaussian distribution with mean and variance approximated by $$\boldsymbol{\mu}_\theta(x_t, t)$$ and $$\boldsymbol{\Sigma}_\theta(x_t, t)$$. Note that both $$\boldsymbol{\mu}_\theta$$ and $$\boldsymbol{\Sigma}_\theta(x_t, t)$$ are conditioned on $$t$$, considered as a positional encoding or *timestep guidance*.

### Loss function

#### Maximum Likelihood Learning.

DDPM\citep{ddpm} objective function also minimizes the "closeness" between data distribution $$\log p_{\rm data}(x_0)$$ and empirical estimated distribution $$p_\theta(x_0)$$ as VAEs\citep{vae}, by maximizing the estimated entropy or negative log-likelihood (refers to Equation \eqref{eq:nlll}), which is detailedly derived at Section \ref{sec:mll}.

$$\begin{align}\label{eq:nlll}
    \mathcal{L}(x_0) = -\mathbb{E}_{x_0\sim p_{\rm data}(x_0)}\log p_\theta(x_0)
\end{align}$$

To sample the data through a Markov chain process, where each transition is a Gaussian distribution, the loss function needs to minimize the distribution distance between two transitions, which are *forward transition* and *reverse transition*. To minimize the distribution divergence between two transitions, DDPM\citep{ddpm} uses KL-divergence as follows:

$$\begin{align}
    \mathcal{L}(q(x_{1:T}|x_0), p_\theta(x_{1:T}|x_0)) = \mathcal{D}_{\rm KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0))
\end{align}$$

Given that KL-divergence is a positive value function, the upper bound of the negative log-likelihood is then obtained as follows:

$$\begin{align}
    0 &\leq \mathcal{D}_{\rm KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0)) \\
    -\log p_\theta(x_0) &\leq -\log p_\theta(x_0) + \mathcal{D}_{\rm KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0))
\end{align}$$

#### Variational Lower Bound.

Based on VAE\citep{vae} setup, the loss function is the negative log-likelihood optimized by a variational lower bound (refers to Equation \eqref{eq:diff-lvb}). The entropy of the input data $$\mathcal{H}(x_0) = -\log p_\theta(x_0)$$ now has the upper bound, presented in the last equal sign. The reverse process loss function then minimizes this upper bound (variational lower bound can be proven using Jensen's equality at Section \ref{sec:vlb-jensen}). 

$$\begin{align}\label{eq:diff-lvb}
    -\log p_\theta(x_0) &\leq -\log p_\theta(x_0) + \mathcal{D}_{\rm KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0))  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)}  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T},x_0)/p_\theta(x_0)}  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})/p_\theta(x_0)}  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\log\left[\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}p_\theta(x_0)\right]  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0)\right]  \\
    &=-\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right] + \log p_\theta(x_0)  \\
    &=\mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]
\end{align}$$

#### Training Loss Function $$\mathcal{L}_{\rm VLB}$$.

From Equation \eqref{eq:diff-lvb}, the training loss function is then expressed as follows:

$$\begin{align}\label{eq:vlb-upper}
    \mathcal{L}_{\rm VLB} &= \mathbb{E}_{q(x_{0:T})}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]  \\
    &= \mathbb{E}_{q(x_{0:T})}\left[\log\frac{\textcolor{blue}{\prod_{t=1}^T q(x_t|x_{t-1})}}{\textcolor{red}{p_\theta(x_T)}\textcolor{blue}{\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)}}\right]
\end{align}$$

Based on Markov chain properties, the nominator and dominator of Equation \eqref{eq:vlb-upper} can be expressed as the product of conditional probabilities (The detailed proofs are at Sections \eqref{eq:vlb-proof:upper} and \eqref{eq:vlb-proof:lower}). Equation \eqref{eq:vlb-upper} can be expressed as follows:

$$\begin{align}
    \mathcal{L}_{\rm VLB} &= \mathbb{E}_{q(x_{0:T})}\left[\log\frac{1}{\textcolor{red}{p_\theta(x_T)}} + \log\textcolor{blue}{\frac{\prod_{t=1}^T q(x_t\mid x_{t-1})}{\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)}}\right]  \\
    &= \mathbb{E}_{q(x_{0:T})}\left[-\log p_\theta(x_T) + \sum_{t=1}^T\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}\right]  \\
    &= \mathbb{E}_{q(x_{0:T})}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\frac{\textcolor{teal}{q(x_t\mid x_{t-1})}}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]
\end{align}$$

Applying the Bayes' Rule, the nominator can be expressed as follows:

$$\begin{align}
    \textcolor{teal}{q(x_t \mid x_{t-1})} = \frac{q(x_t, x_{t-1})}{q(x_{t-1})} = \frac{q(x_{t-1}\mid x_t)q(x_t)}{q(x_{t-1})}
\end{align}$$

$$\begin{align}\label{eq:vlb-norm-x0}
    \textcolor{teal}{q(x_t \mid x_{t-1}, x_0)} = \frac{q(x_t, x_{t-1}, x_0)}{q(x_{t-1}, x_0)} = \frac{q(x_{t-1}\mid x_t, x_0)q(x_t, x_0)}{q(x_{t-1}, x_0)}
\end{align}$$

According to Denoising Diffusion Probabilistic Models \citep{ddpm}, $$\textcolor{teal}{q(x_t \mid x_{t-1})}$$ is intractable, hence conditioning it with $$x_0$$. Note that conditioning on $$x_0$$ does not break the Bayes' Rule (refers to Equation \eqref{eq:vlb-norm-x0}). The loss function $$\mathcal{L}_{VLB}$$ can be expressed as follows:

$$\begin{align}
    \mathcal{L}_{\rm VLB} &= \mathbb{E}_{q(x_{0:T})}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t)}{p_\theta(x_{t-1}\mid x_t)}\frac{q(x_t)}{q(x_{t-1})} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  \\
    &= \mathbb{E}_{q}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)}\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  \\
    &= \mathbb{E}_{q}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\left(\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}\right) + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  \\
    &= \mathbb{E}_{q}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\left(\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}\right) + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  \\
    &= \mathbb{E}_{q}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} + \textcolor{red}{\sum_{t=2}^T\log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  
\end{align}$$

The second sum of logarithms can be shortened as follows:

$$\begin{align}
    \textrm{\textcolor{red}{$\sum_{t=2}^T\log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}$}} &= \log\prod_{t=2}^T \frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}  \\
    &= \log\frac{q(x_2\mid x_0)}{q(x_1\mid x_0)}\frac{q(x_3\mid x_0)}{q(x_2\mid x_0)}\dots\frac{q(x_{T-1}\mid x_0)}{q(x_{T-2}\mid x_0)}\frac{q(x_T\mid x_0)}{q(x_{T-1}\mid x_0)}  \\
    &= \textcolor{blue}{\log\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}}
\end{align}$$

Substituting $$\sum_{t=2}^T\log(q(x_t\mid x_0)/q(x_{t-1}\mid x_0)) = \log(q(x_T\mid x_0)/q(x_1\mid x_0))$$, we have:

$$\begin{align}
    \mathcal{L}_{\rm VLB} &= \mathbb{E}_{q}\left[-\log p_\theta(x_T) + \sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} + \textrm{\textcolor{blue}{$\log\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}$}} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\right]  \\
    &= \mathbb{E}_{q}\left[\log\frac{q(x_T\mid x_0)}{p_\theta(x_T)} + \sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} -\log(p_\theta(x_0\mid x_1))\right]  \\
    &= \mathbb{E}_{q}\left[\underbrace{\mathcal{D}_{\rm KL}(q(x_T\mid x_0) \| p_\theta(x_T))}_{\mathcal{L}_T}+ \sum_{t=2}^T\underbrace{\mathcal{D}_{\rm KL}(\textcolor{red}{q(x_{t-1} \mid x_t, x_0)})\|p_\theta(x_{t-1}\mid x_t))}_{\mathcal{L}_{t-1}}-\underbrace{\log(p_\theta(x_0\mid x_1))}_{\mathcal{L}_0}\right] \label{eq:final-general-vlb}
\end{align}$$

#### Conditional Forward Transition Estimation.

It is noteworthy that the reverse conditional probability is tractable when conditioned on $$x_0$$ (refers to Equation \eqref{eq:vlb-norm-x0}). To estimate \textrm{\textcolor{red}{$$q(x_{t-1} \mid x_t, x_0)$$}}, \citep{ddpm} considers the probability as Gaussian distribution and tried to find $$\boldsymbol{\Tilde{\mu}}(x_{t-1}, x_0)$$ and $$\Tilde{\beta}_t$$.

$$\begin{align}
    q(x_{t-1} \mid x_t, x_0) \sim\mathcal{N}(x_{t-1}; \boldsymbol{\Tilde{\mu}}(x_t, x_0), \Tilde{\beta}_t\boldsymbol{I})
\end{align}$$

Applying the Bayes' Rule and Markov chain, $$q(x_t\mid x_{t-1}, x_0)$$ can be expressed as follows:

$$\begin{align}
    \mathcal{I} = q(x_{t-1} \mid x_t, x_0) = q(x_t\mid x_{t-1}, x_0)\frac{q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)} = q(x_t\mid x_{t-1})\frac{q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
\end{align}$$

Given that the *forward process* is based on Gaussian distribution (refers to Equation ), hence $$\boldsymbol{\Tilde{\mu}}(x_{t-1}, x_0)$$ and $$\Tilde{\beta}_t$$ can be found by applying the probability distribution function.

$$\begin{align}
    \begin{cases}
        q(x_t\mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{a_t}x_{t-1}, (1 - a_t)\textbf{I}) = (2\pi(1-a_t))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_t - \sqrt{a_t}x_{t-1})^2}{-2(1-a_t)}\Big\}}\\
        q(x_{t-1}\mid x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{a}_{t-1}}x_0, (1 - \bar{a}_{t-1})\textbf{I}) = (2\pi(1-\bar{a}_{t-1}))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_{t-1} - \sqrt{\bar{a}_{t-1}}x_0)^2}{-2(1-\bar{a}_{t-1})}\Big\}}\\
        q(x_t\mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{a}_t}x_0, (1 - \bar{a}_t)\textbf{I}) = (2\pi(1-\bar{a}_t))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_t - \sqrt{\bar{a}_t}x_0)^2}{-2(1-\bar{a}_t)}\Big\}}
    \end{cases}
\end{align}$$

Substituting $$q(x_t\mid x_{t-1})$$, $$q(x_{t-1}\mid x_0)$$, and $$q(x_t\mid x_0)$$ into $$q(x_t\mid x_{t-1}, x_0)$$, we have

$$\begin{align}
    \mathcal{I} &= (2\pi(1-a_t))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_t - \sqrt{a_t}x_{t-1})^2}{-2(1-a_t)}\Big\}}\frac{(2\pi(1-\bar{a}_{t-1}))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_{t-1} - \sqrt{\bar{a}_{t-1}}x_0)^2}{-2(1-\bar{a}_{t-1})}\Big\}}}{(2\pi(1-a_t))^{-\frac{1}{2}}\exp{\Big\{\frac{(x_t - \sqrt{a_t}x_0)^2}{-2(1-\bar{a}_t)}\Big\}}} \\
    &\varpropto \exp{\left\{\frac{(x_t - \sqrt{a_t}x_{t-1})^2}{-2(1-a_t)}\right\}}\frac{\exp{\left\{\frac{(x_{t-1} - \sqrt{\bar{a}_{t-1}}x_0)^2}{-2(1-\bar{a}_{t-1})}\right\}}}{\exp{\left\{\frac{(x_t - \sqrt{a_t}x_0)^2}{-2(1-\bar{a}_t)}\right\}}} \\
    &\varpropto \exp{\left\{\frac{(x_t - \sqrt{a_t}x_{t-1})^2}{(1-a_t)} + \frac{(x_{t-1} - \sqrt{\bar{a}_{t-1}}x_0)^2}{(1-\bar{a}_{t-1})} - \frac{(x_t - \sqrt{a_t}x_0)^2}{(1-\bar{a}_t)}\right\}} \\
    &= \exp{\left\{\frac{x_t^2 - 2\sqrt{a_t}x_t x_{t-1} + a_t x_{t-1}^2}{1-a_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{a}_{t-1}}x_0 x_{t-1} + \bar{a}_{t-1}x_0^2}{1 - \bar{a}_{t-1}} - \frac{(x_t - \sqrt{a_t}x_0)^2}{(1-\bar{a}_t)}\right\}}\\
    &= \exp{\left\{\frac{-2\sqrt{a_t}x_t x_{t-1} + a_t x_{t-1}^2}{1-a_t} + \frac{ x_{t-1}^2 -2\sqrt{\bar{a}_{t-1}}x_0 x_{t-1}}{1 - \bar{a}_{t-1}} - \frac{(x_t - \sqrt{a_t}x_0)^2}{(1-\bar{a}_t)} + \frac{x_t^2}{1-a_t} + \frac{\bar{a}_{t-1}x_0^2}{1-\bar{a}_{t-1}}\right\}}\\
    &= \exp{\left\{\frac{-2\sqrt{a_t}x_t x_{t-1} + a_t x_{t-1}^2}{1-a_t} + \frac{x_{t-1}^2 -2\sqrt{\bar{a}_{t-1}}x_0 x_{t-1}}{1 - \bar{a}_{t-1}} + \mathcal{C}(x_t, x_0)\right\}}
\end{align}$$

where $$\mathcal{C}(x_t, x_0) = \frac{(x_t - \sqrt{a_t}x_0)^2}{(1-\bar{a}_t)} + \frac{x_t^2}{1-a_t} + \frac{\bar{a}_{t-1}x_0^2}{1-\bar{a}_{t-1}}$$. Since $$\mathcal{C}(x_t, x_0)$$ does not contain $$x_{t-1}$$, thus omitting it. 

$$\begin{align}
    \mathcal{I} &= \exp{\left\{\frac{-2\sqrt{a_t}x_t x_{t-1} + a_t x_{t-1}^2}{1-a_t} + 
    \frac{x_{t-1}^2 -2\sqrt{\bar{a}_{t-1}}x_0 x_{t-1}}{1 - \bar{a}_{t-1}}
    \right\}}\\
    &= \exp{\left\{\frac{-2\sqrt{a_t}x_t}{1-a_t} x_{t-1} + \frac{a_t}{1-a_t}x_{t-1}^2 + \frac{1}{1-\bar{a}_{t-1}}x_{t-1}^2 - \frac{2\sqrt{\bar{a}_{t-1}}x_0}{1-\bar{a}_{t-1}}x_{t-1}\right\}}\\
    &= \exp{\left\{\Big(\frac{a_t}{1-a_t} + \frac{1}{1-\bar{a}_{t-1}}\Big)x_{t-1}^2 -2\Big(\frac{\sqrt{a_t}x_t}{1-a_t} + \frac{\sqrt{\bar{a}_{t-1}}x_0}{1-\bar{a}_{t-1}}\Big)x_{t-1}\right\}}
\end{align}$$

Given that the second-order term $$x_{t-1}^2$$ and first-order term $$x_{t-1}$$ are related to variance and mean part of Gaussian Distribution (refers to Equation \eqref{eq:uni-gauss}). Therefore $$\boldsymbol{\Tilde{\mu}}(x_t, x_0)$$ and $$\Tilde{\beta}_t\boldsymbol{I}$$ are obtained as follows:

$$\begin{align}
    \Tilde{\beta}_t &= 1/\Big(\frac{a_t}{1-a_t} + \frac{1}{1-\bar{a}_{t-1}}\Big) = 1/\Big(\frac{a_t}{\beta_t} + \frac{1}{1-\bar{a}_{t-1}}\Big) = 1/\Big(\frac{(a_t)(1-\bar{a}_{t-1}) + \beta_t}{\beta_t(1-\bar{a}_{t-1})}\Big)\\
    &= \frac{\beta_t(1-\bar{a}_{t-1})}{(a_t)(1-\bar{a}_{t-1}) + \beta_t} = \frac{\beta_t(1-\bar{a}_{t-1})}{a_t - a_t\bar{a}_{t-1} + \beta_t} = \frac{\beta_t(1-\bar{a}_{t-1})}{a_t - \bar{a}_t + \beta_t} = \frac{\beta_t(1-\bar{a}_{t-1})}{a_t - \bar{a}_t + 1 - a_t}\\
    &= \frac{1-\bar{a}_{t-1}}{1 - \bar{a}_t}\beta_t
\end{align}$$

Applying Equation \eqref{eq:uni-gauss-transition}, $$\boldsymbol{\Tilde{\mu}}(x_t, x_0)$$ can be obtained as follows:

$$\begin{align}
    \boldsymbol{\Tilde{\mu}}(x_t, x_0) &= \Big(\frac{\sqrt{a_t}x_t}{1-a_t} + \frac{\sqrt{\bar{a}_{t-1}}x_0}{1-\bar{a}_{t-1}}\Big)\Tilde{\beta}_t \\
    &= \Big(\frac{\sqrt{a_t}x_t}{1-a_t} + \frac{\sqrt{\bar{a}_{t-1}}x_0}{1-\bar{a}_{t-1}}\Big)\frac{1-\bar{a}_{t-1}}{1 - \bar{a}_t}\beta_t \\
    &= \frac{\sqrt{a_t}x_t}{1-a_t}\frac{1-\bar{a}_{t-1}}{1 - \bar{a}_t}\beta_t + \frac{\sqrt{\bar{a}_{t-1}}x_0}{1-\bar{a}_{t-1}}\frac{1-\bar{a}_{t-1}}{1 - \bar{a}_t}\beta_t \\
    &= \frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1 - \bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1 - \bar{a}_t}x_0 \label{eq:mu-tidle-3}
\end{align}$$

Recall that $$x_t = \sqrt{\bar{a}_t}x_0 + \sqrt{1 - \bar{a}_t}$$, hence $$x_0 = \frac{1}{\sqrt{\bar{a}_t}}(x_t - \sqrt{1 - \bar{a}_t}\epsilon_t)$$. Substituting into Equation \eqref{eq:mu-tidle-3}, we have:

$$\begin{align}
    \boldsymbol{\Tilde{\mu}}(x_t, x_0) &=\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1 - \bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1 - \bar{a}_t}\frac{1}{\sqrt{\bar{a}_t}}(x_t - \sqrt{1 - \bar{a}_t}\epsilon_t) \\
    &=\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1 - \bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{\sqrt{\bar{a}_t}(1 - \bar{a}_t)}x_t - \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{\sqrt{(1-\bar{a}_t)\bar{a}_t}}\epsilon_t \\
    &=\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1 - \bar{a}_t}x_t + \frac{\beta_t}{\sqrt{a_t}(1 - \bar{a}_t)}x_t - \frac{\beta_t}{\sqrt{(1-\bar{a}_t)a_t}}\epsilon_t \\
    &=\frac{a_t(1-\bar{a}_{t-1})}{\sqrt{a_t}(1 - \bar{a}_t)}x_t + \frac{\beta_t}{\sqrt{a_t}(1 - \bar{a}_t)}x_t - \frac{\beta_t}{\sqrt{(1-\bar{a}_t)a_t}}\epsilon_t\\
    &=\frac{1}{\sqrt{a_t}}\Big(\frac{a_t - a_t\bar{a}_{t-1} + 1 - a_t}{1 - \bar{a}_t}x_t - \frac{1-a_t}{\sqrt{(1-\bar{a}_t)}}\epsilon_t\Big) \\
    &=\frac{1}{\sqrt{a_t}}\Big(\frac{1 - a_t\bar{a}_{t-1}}{1 - \bar{a}_t}x_t - \frac{1-a_t}{\sqrt{(1-\bar{a}_t)}}\epsilon_t\Big) \\
    &=\frac{1}{\sqrt{a_t}}\Big(x_t - \frac{1-a_t}{\sqrt{(1-\bar{a}_t)}}\epsilon_t\Big)\label{eq:final-mu-tidle}
\end{align}$$

#### Loss function parameterization.

Recall that a neural network needs training to approximate the conditional probability in the *reverse process* $$p_\theta(x_{t-1}\mid x_{t}) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)$$. Therefore, model $$\mu_\theta$$ needs training to predict $$\mu_t = \frac{1}{\sqrt{a_t}}(x_t - \frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\epsilon_t)$$ (refers to Equation \eqref{eq:final-mu-tidle}). 

$$\begin{align}
    \boldsymbol{\Tilde{\mu}}_\theta(x_t, t) &= \frac{1}{\sqrt{a_t}}\left(x_t - \frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\epsilon_\theta(x_t, t)\right) \\
    \textrm{Thus $x_{t-1}$} &\sim \mathcal{N}(x_{t-1}, \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) \\
     &\sim \mathcal{N}(x_{t-1}, \frac{1}{\sqrt{a_t}}(x_t - \frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\epsilon_\theta(x_t, t)), \Sigma_\theta(x_t, t))
\end{align}$$

In DDPM\citep{ddpm}, to untrained time dependent constants, $$\Sigma_\theta(x_t, t)$$ is set to $$\sigma_t^2\textbf{I}$$ (aforementioned in Section \ref{sec:foward}). Experimentally, setting $$\sigma_t^2 = \beta_t$$ or $$\sigma_t^2 = \Tilde{\beta}_t$$ gives the same result, while the first choice is optimal for $$x_0\sim\mathcal{N}(0, \textbf{I})$$, the second choice is optimal for $$x_0$$ is deterministically set to one point\citep{ddpm}. From Equation \eqref{eq:final-general-vlb}, the parameterized loss function can be derived as follows:

$$\begin{align}
    \mathcal{L}_{t-1} &= \mathcal{D}_{\rm KL}(q(x_{t-1} \mid x_t, x_0))\|p_\theta(x_{t-1}\mid x_t)) \\
    &= \mathbb{E}_{x_0, \epsilon}\left[\log(q(x_{t-1} \mid x_t, x_0)) - \log(p_\theta(x_{t-1}\mid x_t))\right] 
\end{align}$$

Since, $$q(x_{t-1} \mid x_t, x_0)$$ and $$p_\theta(x_{t-1}\mid x_t)$$ are both Gaussian distributions, hence the KL-divergence can be obtained by applying the closed form (refers to Equations \eqref{eq:cf-gen-gauss} and \eqref{eq:cf-svar-gauss}). The KL-divergence can be obtained as follows:

$$\begin{align}
    \mathcal{L}_{t-1} &= \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\|\boldsymbol{\Tilde{\mu}}(x_t, x_0) - \boldsymbol{\mu}_\theta(x_t, t)\|^2\right] + C \\
    \Leftrightarrow \mathcal{L}_{t-1} &= \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{a_t}}\left(x_t - \frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\epsilon_t\right) - \frac{1}{\sqrt{a_t}}\left(x_t - \frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\epsilon_\theta(x_t, t)\right)\right\|^2\right] + C \\
    &= \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\left\|\epsilon_t - \epsilon_\theta(x_t, t)\right\|^2\right] + C \\
    &= \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\frac{1-a_t}{\sqrt{1 - \bar{a}_t}}\left\|\epsilon_t - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1- \bar\alpha_t}\epsilon_t, t)\right\|^2\right] + C 
\end{align}$$

Authors of DDPM\citep{ddpm} found out that it beneficial to sample quality (and simpler to implement) to train on the following variant of the variational bound:

$$\begin{align}
    \mathcal{L}_{t-1}(\theta) \propto \mathcal{L}_{\rm simple}(\theta) = \mathbb{E}_{t\sim[1, T]x_0, \epsilon}\left[\left\|\epsilon_t - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1- \bar\alpha_t}\epsilon_t, t)\right\|^2\right]
\end{align}$$