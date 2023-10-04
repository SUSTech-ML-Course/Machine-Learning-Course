# CS405 Homework #1

*Course: Machine Learning(CS405) - Instructor: Qi Hao*

## Question 1

Consider the polynomial function:

$$y(x,\mathbf{w})=w_{0}+w_{1}x+w_{2}x+...+w_Mx^M=\sum^M_{i=0}w_ix^i$$

Calculate the coefficients $\mathbf{w}=\{w_i\}$ that minimize its sum-of-squares error function. Here a suffix $i$ denotes the index of a component, whereas $(x)^i$ denotes $x$ raised to the power of $i$.



## Question 2

Suppose that we have three colored boxes $r(\mathrm{red})$, $b(\mathrm{blue})$, and $g(\mathrm{green})$. Box $r$ contains 3 apples, 4 oranges, and 3 limes, box $b$ contains 1 apple, 1 orange, and 0 limes, and box $g$ contains 3 apples, 3 oranges, and 4 limes. If a box is chosen at random with probabilities $p(r)=0.2, p(b) = 0.2, p(g) = 0.6$, and a piece of fruit is removed from the box (with equal probability of selecting any of the items in the box), then what is the probability of selecting an apple? If we observe that the selected fruit is in fact an orange, what is the probability that it came from the green box?



## Question 3

Given two statistically independent variables $x$ and $z$, show that the mean and variance of their sum satisfies

$\mathbb{E}[x+z]=\mathbb{E}[x]+\mathbb{E}[z]$

$\mathrm{var}[x+z] = \mathrm{var}[x]+\mathrm{var}[z]$



## Question 4

In probability theory and statistics, the Poisson distribution, is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event. If ${X}$ is Poisson distributed, i.e. $X\sim Possion(\lambda)$, its probability mass function takes the following form: 

$P(X|\lambda)=\frac{\lambda^Xe^{-\lambda}}{X!}$

It can be shown that if $\mathbb{E}(X) = \lambda$. Assume now we have $n$ data points from $Possion(\lambda): \mathcal{D}=\{X_1, X_2,..., X_n\}$. Show that the sample mean $\widehat{\lambda}=\frac{1}{n}\sum^n_{i=1}X_i$ is the maximum likelihood estimate(MLE) of $\lambda$.

If $X$ is exponential distribution and its distribution density function is $f(x)=\frac{1}{\lambda}e^{-\frac{x}{\lambda}}$ for $x>0$ and $f(x)=0$ for $x\leq0$. Show that the sample mean $\widehat{\lambda}\frac{1}{n}\sum^n_{i=1}X_i$ is the maximum likelihood estimate(MLE) of $\lambda$.



## Question 5

*(a)* Write down the probability of classifying correctly $p(correct)$ and the probability of misclassification $p(mistake)$ according to the following chart.

<img src="HW1.assets/mistake.jpg" alt="mistake" style="zoom:50%;" />

*(b)* For multiple target variables described by vector $\mathbf{t}$, the expected squared loss function is given by

$\mathbb{E}[\mathit{L}\mathbf{(t, y(x))}]=\int\int \left \| \mathbf{y(x)-t} \right \|^2p(\mathbf{x, t})\mathrm{d}\mathbf{x}\mathrm{d}\mathbf{t}$

Show that the function $\mathbf{y(x)}$ for which this expected loss is minimized given by $\mathbf{y(x)}=\mathbb{E}\mathbf{_t[t|x]}$.

> #### Hints
>
> For a single target variable $t$, the loss is given by
>
> $\mathbb{E}[\mathit{L}]=\int\int\{y(\mathbf{x})-t\}^2p(\mathbf{x}, t)\mathrm{d}\mathbf{x}\mathrm{dt}$
>
> The result is as follows
>
> $y(\mathbf{x})=\frac{\int tp(\mathbf{x}, t)\mathrm{dt}}{p(\mathbf{x})}=\int tp(t|\mathbf{x})\mathrm{dt}=\mathbb{E}_t[t|\mathbf{x}]$



## Question 6

*(a)* We defined the entropy based on a discrete random variable $\mathbf{X}$ as

$\mathbf{H[X]}=-\sum_{i}p(x_i)\mathrm{ln} p(x_i)$

Now consider the case that $\mathbf{X}$ is a continuous random variable with the probability density function $p(x)$. The entropy is defined as

$\mathbf{H[X]}=-\int p(x)\mathrm{ln} p(x) dx$

Assume that $\mathbf{X}$ follows Gaussian distribution with the mean $\mu$ and variance $\sigma$, i.e.

$p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

Please derive its entropy $\mathbf{H[X]}$.

*(b)* Write down the mutual information $\mathbf{I(y, x)}$. Then show the following equation

$\mathbf{I[x,y]=H[x]-H[x|y]=H[y]-H[y|x]}$

