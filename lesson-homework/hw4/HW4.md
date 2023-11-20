# Homework Ⅳ

*Course: Machine Learning(CS405) - Professor: Qi Hao*

## Question 1

Show that maximization of the class separation criterion given by $m_2 - m_1 = \mathbf{w^T(m_2 - m_1)}$ with respect to $\mathbf w$, using a Lagrange multiplier to enforce the constraint $\mathbf{w^T w = 1}$, leads to the result that $\mathbf w \propto \mathbf{(m_2 - m_1)}$.

## Question 2

Show that the Fisher criterion

$$
\mathrm J(\mathbf w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}
$$

can be written in the form

$$
\mathrm J(\mathbf w) = \mathbf{\frac{w^T S_B w}{w^T S_W w}}
$$

**Hint.**

$$
y = \mathbf{w^T x},\qquad
$$

$$
m_k = \mathbf{w^T m_k},\qquad
$$

$$
s_k^2 = \sum_{n\in\mathcal C_k}(y_n - m_k)^2
$$

## Question 3

Consider a generative classification model for $K$ classes defined by prior class probabilities $p(\mathcal C_k) = \pi_k$ and general class-conditional densities $p(\phi|\mathcal C_k)$ where $\phi$ is the input feature vector. Suppose we are given a training data set \{ $\phi_n, \mathbf t_n$ \} where $n = 1, ..., N$, and $\mathbf t_n$ is a binary target vector of length $K$ that uses the 1-of-K coding scheme, so that it has components $t_{nj} = I_{jk}$ if pattern $n$ is from class $\mathcal C_k$.

Assuming that the data points are drawn independently from this model, show that the maximum-likelihood solution for the prior probabilities is given by

$$
\pi_k = \frac{N_k}{N},
$$

where $N_k$ is the number of data points assigned to class $\mathcal C_k$.

## Question 4

Verify the relation

$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

for the derivative of the logistic sigmoid function defined by

$$
\sigma(a) = \frac{1}{1 + \mathrm{exp}(-a)}
$$

## Question 5

By making use of the result

$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

for the derivative of the logistic sigmoid, show that the derivative of the error function for the logistic regression model is given by

$$
\nabla \mathbb E(\mathbf w) = \sum^N_{n=1}(y_n - t_n)\phi_n.
$$

**Hint.**

The error function for the logistic regression model is given by

$$
\mathbb E(\mathbf w) = -\mathrm{ln}p(\mathbf{t|w}) = -\sum^N_{n=1}\{t_n\mathrm{ln}y_n + (1 - t_n)\mathrm{ln}(1 - y_n)\}.
$$

## Question 6

There are several possible ways in which to generalize the concept of linear discriminant functions from two classes to $c$ classes. One possibility would be to use ( $c-1$ ) linear discriminant functions, such that $y_k(\mathbf x )>0$ for inputs $\mathbf{x}$ in class $C_k$ and $y_k(\mathbf{x})<0$ for inputs not in class $C_k$.

By drawing a simple example in two dimensions for $c = 3$, show that this approach can lead to regions of x-space for which the classification is ambiguous.

Another approach would be to use one discriminant function $y_{jk}(\mathbf{x})$ for each possible pair of classes $C_j$ and $C_k$ , such that $y_{jk}(\mathbf{x})>0$ for patterns in class $C_j$ and $y_{jk}(\mathbf{x})<0$ for patterns in class $C_k$. For $c$ classes, we would need $c(c-1)/2$ discriminant functions.

Again, by drawing a specific example in two dimensions for $c = 3$, show that this approach can also lead to ambiguous regions.

## Question 7

Given a set of data points { $\{\mathbf{x}^n\}$ } we can define the convex hull to be the set of points $\mathbf{x}$ given by

$$
\mathbf{x} = \sum_n\alpha_n\mathbf{x}^n
$$

where $\alpha_n>=0$ and $\sum_n\alpha_n=1$. Consider a second set of points $\{\mathbf{z}^m\}$ and its corresponding convex hull. The two sets of points will be linearly separable if there exists a vector $\hat{\mathbf{w}}$ and a scalar $w_0$ such that $\hat{\mathbf{w}}^T\mathbf{x}^n+w_0>0$ for all $\mathbf{x}^n$, and $\hat{\mathbf{w}}^T\mathbf{z}^m+w_0<0$ for all $\mathbf{z}^m$.

Show that, if their convex hulls intersect, the two sets of points cannot be linearly separable, and conversely that, if they are linearly separable, their convex hulls do not intersect.
