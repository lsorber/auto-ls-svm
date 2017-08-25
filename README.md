# Auto LS-SVM

_This is a work in progress._

## Goals

Auto SVM is a Least Squares Support Vector Machine (LS-SVM) that is improved in several ways:

1. A next-gen regularization term that penalizes model complexity directly:
    - Regression: the objective is a tradeoff between fit on training data and a direct estimation of model complexity (instead of a proxy like the L2 norm).
    - Classification: the objective is a mix between fit on the training data, model complexity (next-gen term) and maximizing the margin (L2 norm). While maximizing the margin is important, minimizing the decision boundary complexity may significantly improve generalization performance.
2. Fully normalized so that:
    - Hyperparameters are easily interpreted.
    - Default hyperparameter values of $1.0$ will give good results.
    - Adding or removing rows (observations) or columns (features) has a minimal influence of the choice of the hyperparameter values.
3. Least squares formulation allows for a cheap closed-form expression for the leave-one-out error. This enables a powerful way to search for the optimal hyperparameters.
4. In a later stage, perhaps a large scale implementation too.

## A next-gen regularization term

We begin with an observation $x$, and dual variables $\alpha$:
$$
x, x_i \in \mathbb{R}^{d \times 1}\\
\alpha \in \mathbb{R}^{n \times 1}\\
X := \left[x_1 ~ \ldots ~ x_n\right] \in \mathbb{R}^{d \times n}
$$

We choose an RBF kernel in the same format as `sklearn`:
$$
\begin{align}
k(x,y) &:= \exp(-\gamma \|x-y\|^2) \in \mathbb{R}\\
K(x) &:= \left[k(x, x_1) ~ \ldots ~ k(x, x_n)\right]  \in \mathbb{R}^{1 \times n}
\end{align}
$$

The gradient of the kernel is given by:
$$
\begin{align}
\frac{dk}{dx} &= -2\gamma k(x,y) \cdot (x-y)\\
\nabla K := \frac{dK}{dx} &= -2\gamma\left( x \cdot 1_{1\times n}  - X \right) \cdot \mathrm{diag}(K(x))
\end{align}
$$

The SVM model is of the form:
$$
f(x) := K(x)\cdot \alpha
$$

$$
\min_{\alpha} \|y - f(x)\|^2 + \mu\cdot \mathrm{nextgen} + \nu\cdot \|\alpha\|^2
$$

The normal on the prediction surface is:
$$
n := \begin{bmatrix}\nabla K \cdot \alpha\\ -1\end{bmatrix}
$$

And so the norm of the normal vector is:
$$
\begin{align}
\|n\|^2 &= \alpha^T\cdot (\nabla K)^T (\nabla K) \cdot \alpha + 1\\
&= 1 + 4\gamma^2 \cdot \alpha^T \cdot \mathrm{diag}(K(x)) \cdot \left(\|x\|^2 - 1_{n\times 1} \cdot x^T \cdot X - X^T \cdot x \cdot 1_{1 \times n} + X^T X\right) \cdot \mathrm{diag}(K(x)) \cdot \alpha\\
&= 1 + 4\gamma^2 \cdot \alpha^T \cdot \left[ k(x,x_i)k(x,x_j)\left(\|x\|^2 - x_i^T\cdot x - x_j^T\cdot x + x_i^T\cdot x_j \right) \right] \cdot \alpha
\end{align}
$$

We can integrate $\|n\|$ over the prediction surface to obtain its $d$-volume, but then we need a definite integral with finite bounds. Instead, we can integrate $\|\nabla K \cdot \alpha\|$ (the norm of the gradient of the prediction surface) and use infinite bounds, which has the effect of simplifying the integral.

### Derivation of the regularization term

Let's integrate each of the three types of terms.

##### Term of the form $k(x,x_i)k(x,x_j)x_i^T\cdot x_j$

First, let's integrate out the $p$-th dimension of $x$:

$$
\begin{align}
I^{(3,p)}_{ij} &= \int_{-\infty}^\infty k(x,x_i)k(x,x_j)x_i^T\cdot x_j dx^{(p)}\\
&= k(x^{(/p)}, x_i^{(/p)}) k(x^{(/p)}, x_j^{(/p)})x_i^T\cdot x_j  \int_{-\infty}^\infty \exp(-\gamma (x^{(p)} - x_i^{(p)})^2-\gamma (x^{(p)} - x_j^{(p)})^2) dx^{(p)}\\
&= C_{ij} \int_{-\infty}^\infty \exp(-\gamma (x^{(p)} - x_i^{(p)})^2-\gamma (x^{(p)} - x_j^{(p)})^2) dx^{(p)}\\
&= C_{ij} \int_{-\infty}^\infty \exp\left(-2\gamma\left(x^{(p)} - \frac{x_i^{(p)}+x_j^{(p)}}{2}\right)^2 - 
\frac{\gamma}{2}\left(x_i^{(p)} - x_j^{(p)}\right)^2\right) dx^{(p)}\\
&= C_{ij} \sqrt{\frac{\pi}{2 \gamma}} \exp\left(- 
\frac{\gamma}{2}\left(x_i^{(p)} - x_j^{(p)}\right)^2\right)
\end{align}
$$

This means that after integrating out all $d$ dimensions, we get:

$$
I_{ij}^{(3)} = x_i^T\cdot x_j\left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}}\sqrt{k(x_i,x_j)}
$$

##### Term of the form $k(x,x_i)k(x,x_j)x_i^T\cdot x$

First, let's integrate out the $p$-th dimension ($p \ne q$) of $x$:

$$
\begin{align}
I^{(2,p,i)}_{ij} &= \int_{-\infty}^\infty k(x,x_i)k(x,x_j)x_i^T\cdot x dx^{(p)}\\
&= \sum_{q=1}^d\int_{-\infty}^\infty k(x,x_i)k(x,x_j)x_{i}^{(q)} x^{(q)} dx^{(p)}\\
&= \sum_{q=1}^d  k(x^{(/p)}, x_i^{(/p)}) k(x^{(/p)}, x_j^{(/p)}) x_{i}^{(q)} x^{(q)} \sqrt{\frac{\pi}{2 \gamma}} \exp\left(- 
\frac{\gamma}{2}\left(x_i^{(p)} - x_j^{(p)}\right)^2\right)
\end{align}
$$

Next, we integrate out all other dimensions:

$$
\begin{align}
I_{ij}^{(2,i)} &= \sum_{q=1}^d   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d-1}{2}} \sqrt{k(x_i^{(/q)},x_j^{(/q)})} \int_{-\infty}^{\infty} k\left(x^{(q)}, x_i^{(q)}\right) k\left(x^{(q)}, x_j^{(q)}\right) x_{i}^{(q)} x^{(q)} dx^{(q)}\\
&=   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)} \sum_{q=1}^d x_i^{(q)} \frac{x_i^{(q)}+ x_j^{(q)}}{2} \\
&=   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)}  x_i \cdot (x_i + x_j) / 2 
\end{align}
$$

##### Term of the form $k(x,x_i)k(x,x_j)\|x\|^2$

First, let's integrate out the $p$-th dimension ($p \ne q$) of $x$:

$$
\begin{align}
I^{(1,p)}_{ij} &= \int_{-\infty}^\infty k(x,x_i)k(x,x_j) \|x\|^2 dx^{(p)}\\
&= \sum_{q=1}^d\int_{-\infty}^\infty k(x,x_i)k(x,x_j)x^{(q)2} dx^{(p)}\\
&= \sum_{q=1}^d  k(x^{(/p)}, x_i^{(/p)}) k(x^{(/p)}, x_j^{(/p)}) x^{(q)2} \sqrt{\frac{\pi}{2 \gamma}} \exp\left(- 
\frac{\gamma}{2}\left(x_i^{(p)} - x_j^{(p)}\right)^2\right)
\end{align}
$$

Next, we integrate out all other dimensions:

$$
\begin{align}
I_{ij}^{(1)} &= \sum_{q=1}^d   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d-1}{2}} \sqrt{k(x_i^{(/q)},x_j^{(/q)})} \int_{-\infty}^{\infty} k\left(x^{(q)}, x_i^{(q)}\right) k\left(x^{(q)}, x_j^{(q)}\right) x^{(q)2} dx^{(q)}\\
&=   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)} \sum_{q=1}^d \frac{1}{4}\left((x_i^{(q)}+  x_j^{(q)})^2 + \frac{1}{\gamma}\right) \\
&=   \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)}  \frac{1}{4} \left(\|x_i + x_j\|^2 + \frac{d}{\gamma}\right)
\end{align}
$$

##### Summing the terms up:

$$
\begin{align}
I &= I_{ij}^{(1)}-I_{ij}^{(2,i)}-I_{ij}^{(2,j)}+I_{ij}^{(3)}\\
&= \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)}  \frac{1}{4} \left(\|x_i + x_j\|^2 + \frac{d}{\gamma}\right)\\
&- \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)}  x_i \cdot (x_i + x_j) / 2\\
&- \left(\frac{\pi}{2 \gamma}\right)^{\frac{d}{2}} \sqrt{k(x_i,x_j)}  x_j \cdot (x_i + x_j) / 2 \\
&+ \left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}}\sqrt{k(x_i,x_j)} x_i^T\cdot x_j\\
&= \left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}}\sqrt{k(x_i,x_j)} \left(\frac{1}{4} \left(\|x_i + x_j\|^2 + \frac{d}{\gamma}\right) - x_i \cdot (x_i + x_j) / 2 -  x_j \cdot (x_i + x_j) / 2 + x_i^T\cdot x_j\right)\\
&= \left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}}\sqrt{k(x_i,x_j)} \left(\frac{1}{4} \left(\|x_i + x_j\|^2 + \frac{d}{\gamma}\right) - \frac{1}{2}\|x_i\|^2 - \frac{1}{2}\|x_j\|^2\right)\\
&= \frac{1}{4} \left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}}\sqrt{k(x_i,x_j)} \left( \frac{d}{\gamma} - \|x_i - x_j\|^2\right)
\end{align}
$$

### The resulting formula

$$
\begin{align}
\int \|\nabla f \|^2 &= \int \alpha^T\cdot (\nabla K)^T (\nabla K) \cdot \alpha \\
&= \gamma^2 \left(\frac{\pi}{2\gamma}\right)^{\frac{d}{2}} \cdot \alpha^T \cdot \begin{bmatrix} \sqrt{k(x_i,x_j)} \left( \frac{d}{\gamma} - \|x_i - x_j\|^2\right) \end{bmatrix} \cdot \alpha\\
&\propto \gamma^{-\frac{d}{2}}  \cdot \alpha^T \cdot \begin{bmatrix} \sqrt{k(x_i,x_j)} \left( d\gamma - \gamma^2\|x_i - x_j\|^2\right) \end{bmatrix} \cdot \alpha
\end{align}
$$

## Normalization of the full problem

We want to achieve a number of things:

1. The solution should be close to invariant under addition or removal of columns.
2. The solution should be close to invariant under addition or removal of rows.
3. A default value of $\gamma = 1.0$ should be a good starting value for data sets of any size.
4. A default value of $\mu = 1.0$ should be a good starting value for data sets of any size.

#### Reduction of sensitivity of the regularization term on $d$

First, we fix $d=2$. This way, it is as if we integrated only two dimensions, making the exponential effect of $\gamma$ less problematic.

$$
\alpha^T \cdot \begin{bmatrix} \sqrt{k(x_i,x_j)} \left( 2 - \gamma\|x_i - x_j\|^2\right) \end{bmatrix} \cdot \alpha
$$

~~Second, we replace $\gamma$ with $\frac{\gamma}{d}$ in the kernel function $k$. This makes it so that if you only have one feature ($d=1$) and you add a copy of this feature to the feature matrix ($d=2$), you can keep the same value for $\gamma$ and end up with an identical model.~~

$$
\alpha^T \cdot \begin{bmatrix} \sqrt{k(x_i,x_j)} \left( 2 - \frac{\gamma}{d}\|x_i - x_j\|^2\right) \end{bmatrix} \cdot \alpha
$$

A counterargument to this is that adding an all-zero feature would require a different value of $\gamma$ to reach optimality.
