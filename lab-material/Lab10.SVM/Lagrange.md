$$
\min_{x}f(x)\\
s.t. \ \ h_i(x)=0\ \  (i=1,...,m),\\
\ \ \ \ \ \ g_j(x)\le 0\ \ (j=1,...,n)
$$

Lagrange function:
$$
L(x, \lambda, \mu)=f(x)+\sum_{i=1}^{m}\lambda_ih_i(x)+\sum_{j=1}^{n}\mu_jg_j(x)
$$

KKT condition:
$$
g_j(x)\le 0 \\
\mu_j\ge 0 \\
\mu_jg_j(x)=0
$$

Dual problem:
$$
\Gamma(\lambda,\mu)=\inf_{x\in D}L(x,\lambda,\mu) \le L(\tilde{x},\lambda,\mu) \le f(\tilde{x})
$$

Lagrange function for SVM:
$$
L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^{m}\alpha_i(1-y_i(w^Tx_i+b))
$$

Set above derivative to 0:
$$
w=\sum_{i=1}^{m}\alpha_iy_ix_i \\
0=\sum_{i=1}^{m}\alpha_iy_i
$$

Dual problem:
$$
\max_{\alpha}\sum_{i=1}^{m}\alpha_i-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j
$$