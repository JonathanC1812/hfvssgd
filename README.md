# Background 

Across most of the field, Stochastic Gradient Descent (SGD) and Gradient Descent (GD) are among the most widely used optimization algorithms. In the case of GD, although it is quite efficient, it becomes really slow on ill-conditioned problems, such as narrow-shaped loss surfaces and elongated valleys (i.e, a high difference on the highest eigenvalue compared to the lowest eigenvalue, k). These problems cause GD to zig-zag along the steep direction while slowly crawling along the flat one. On the other hand, SGD is often used as an alternative that can converge faster than GD in these settings, even though it is not a clear-cut best solution either, since label noises could affect its performance. This motivates us to take a gander at the Hessian-Free (HF) method which uses Hessian-vector products that could map out valleys, slopes or saddle-points. Our hypothesis is that although HF requires more computational cost per step, the use of Hessian-vector products could cause it to become more cost-efficient than SGD and GD as k grows larger and even might hold up more consistently across both clean and noisy conditions.

In this experiment, we tested and compared 3 optimization algorithms which are GD, SGD and HF. We want to answer a specific question of whether or not a Hessian-Free Newton method could actually beat plain GD and a small-batch SGD. In Gradient Descent, the convergence rate depends on the condition number of the loss Hessian. The Hessian-Free method instead trades an extra computational cost per-step for a convergence rate that scales with the square-root of the condition number $\sqrt{k}$.

The comparison was performed across a range of condition numbers from a well-conditioned one to a badly-conditioned one. We also checked if it holds once a noise was added to the mix, since one of Newton’s guarantees depends on the loss’s shape rather than the data itself. Finally, We chose a synthetic linear regression testbed approach so that we could monitor the condition number directly rather than measuring it after the fact.

# Repo Structure

For our repo structure, we labeled core.py as the holder of the synthetic data generator with a controllable k, the loss and gradient function and k calculator. On the other hand, we built opt.py to hold all the three optimization algorithms (GD, SGD and HF). opt.py also contains the Hessian-vector product function and CG solver to put all the methods on a common cost scale for a better understanding. Finally, all actual experiments and plots are conducted and contained in sweep.ipynb to facilitate a better showcase of data. 

# Methods

This experiment is specifically done on a **quadratic problem**, so we decide to implement this algorithm for a linear regression neural network.

As we know, the loss for a linear regression model is calculated by

$$
L(w) = \frac{1}{n} \sum_{i=1}^{n} (Xw - y)^2
$$

where $Xw$ is the predicted answer, $y$ is the correct answer, and $n$ is the sample size.

To simplify this equation when we calculate the derivative, we can divide the loss by two, getting

$$
L(w) = \frac{1}{2n} \sum_{i=1}^{n} (Xw - y)^2
$$

where our goal is to minimize this number.

On developing the algorithm we have **3 algorithms**, which are:

---

## 1. Gradient Descent

To implement this algorithm, we first have to calculate the gradient of the loss to find the direction of the descent. From the loss formula given as

$$
L(w) = \frac{1}{2n} \sum_{i=1}^{n} (Xw - y)^2
$$

we are able to find the gradient as

$$
\nabla L(w) = \frac{1}{n} X^{\top} (Xw - y)
$$

Then, the algorithm is performed by

$$
w \leftarrow w - \eta \, \nabla L(w)
$$

where $\eta$ is the learning rate of gradient descent given by

$$
\eta = \frac{2}{\lambda_{\min} + \lambda_{\max}}
$$

until reaching the tolerance required.

## 2. Stochastic Gradient Descent

A stochastic gradient descent is a modified version of gradient descent instead of choosing the direction of descent using the gradient of the entire loss function, we choose a direction and find the derivative according to that specific direction. In this experiment, we use a mini-batch stochastic gradient descent so the direction is decided by choosing a random direction for `batch` amount of direction. For the learning rate, it perform grid-search across `lr_grid` which is a list of a `(0.05, 0.1, 0.25, 0.5, 1.0, 1.5)` with short trial runs per candidate.
$$
\nabla L_B(w) = \frac{1}{b} X_B^{\top} (X_B w - y_B)
$$

This is an unbiased estimate of the full gradient, since $\mathbb{E}[\nabla L_B(w)] = \nabla L(w)$, but it is noisy, so each step is much cheaper while being less accurate. The update is the same shape as gradient descent:

$$
w \leftarrow w - \eta_t \, \nabla L_B(w)
$$

### Choosing the learning rate

Unlike gradient descent, there is no closed form for the best $\eta$ here, so it is tuned by a grid search. For each candidate constant $c$ we try

$$
\eta = \frac{c}{\lambda_{\max}}
$$

run a short trial run of SGD with that value, and keep the $c$ whose run reaches the target loss floor at the lowest cost (or, if none of them reach it, the one with the lowest final loss).

### Decaying the learning rate

Because the mini-batch gradient never becomes exactly zero at the optimum, a constant step size makes the iterates bounce around the minimum instead of settling. To remove that noise floor, the step size is decayed as

$$
\eta_t = \frac{\eta}{1 + \eta \, \lambda_{\min} \, t}
$$

so the steps shrink as $t$ grows and the iterates can converge.


## 3. Hessian-free optimization

Since the problem is a curvature, we could have the solution using the taylor approximation:

$$
f(\theta + p) \approx f(\theta) + \nabla f(\theta)^{\top} p + \tfrac{1}{2} p^{\top} B p
$$

In the Newton's method, $B = H$ or $B = H + \lambda I$ where $H$ is the Hessian matrix being damped. For our project, we will use $B = H$.

To optimize the Newton's method, we can differentiate the function with respect to $p$, and we will get

$$
\nabla_p \left( f(\theta) + \nabla f(\theta)^{\top} p + \tfrac{1}{2} p^{\top} B p \right) = \nabla f(\theta) + B p = 0
$$

which gives

$$
B p = -\nabla f(\theta)
$$

To solve for $p$, which is the optimal next step, we can use conjugate gradient. Lastly, to compute $Bp$ itself, we can manipulate it by calculating

$$
B = \frac{1}{n} X^{\top} X
\quad \Longrightarrow \quad
B p = \frac{1}{n} X^{\top} (X p)
$$

which reduced the complexity from $O(nd^2)$ to $O(nd)$.

Although we will use the hessian matrix, we never have to compute it, hence why it is called a Hessian-"Free" method.

## Benchmarking methods

### 1. Compare noise vs no noise

Every configuration is run twice, once on clean labels ($y = Xw_{\text{true}}$) and once with noise added as

$$
y = X w_{\text{true}} + \sigma \, \operatorname{std}(y) \, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I), \quad \sigma = 0.1
$$

Noise is the falsification test for our hypothesis. It leaves the Hessian $H = X^{\top}X / n$ completely untouched, so the condition number $\kappa$ is identical in both settings, but it raises the optimal loss $L^{*}$ above zero and injects variance into every mini-batch gradient. If HF's advantage really comes from curvature, it should carry over from the clean row to the noisy row unchanged, while SGD should be the one that degrades, since its steps never stop being noisy at the optimum.

### 2. Batch of runs for each cost limits and tolerance targets

Instead of scoring a run by a single end-point, each method records a trace of `(cost, loss)` pairs as it goes, and we slice that trace two different ways:

- **Tolerance target**
fix the tolerance target, measure the cost. The cost at which the run first reaches
  $$
  \frac{L(w) - L^{*}}{L(w_0) - L^{*}} \le \varepsilon
  $$
  for a loose and a tight target, $\varepsilon \in \{10^{-2},\ 10^{-6}\}$.

- **Cost limit** 
    fix the budget, measure the loss reached. 

### 3. Show Two Data: Cost and Time

Our main comparison will be counted by the cost of the algorithm, counted in gradient-equivalent units where one unit is a single $O(nd)$ pass over the data:

$$
\text{cost} = n_{\text{grad}} + n_{\text{hvp}}
$$

GD costs $1$ per step, SGD costs $b/n$ per step, and HF costs $1$ for the gradient of each outer step plus $1$ for every CG iteration inside it, since a Hessian-vector product $Bp = X^{\top}(Xp)/n$ costs the same as a gradient.

The second series is **wall-clock time**, measured per run with `time.perf_counter()`. Cost is a model of runtime, not runtime itself, so the two are reported side by side for benchmark. However, this itself isn't solely reliable because it relies heavily on external factor such as the hardware this program is being run.