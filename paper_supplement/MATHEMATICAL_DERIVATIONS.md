# Detailed Mathematical Derivations

This supplement gives the derivations behind the mathematical statements in the paper *Testing a Shared Low-Dimensional Operator Representation for Time-Varying Financial Dynamics*. It is deliberately self-contained, but it does not introduce a new estimand or strengthen any empirical claim in the paper. Equation and result names below refer to the notation of the paper.

## Table of contents

1. [Notation, conditioning, and timing](#1-notation-conditioning-and-timing)
2. [Linearization of the observation equation](#2-linearization-of-the-observation-equation)
3. [Kalman prediction, updating, and predictive covariance](#3-kalman-prediction-updating-and-predictive-covariance)
   - [State prediction](#31-state-prediction)
   - [Observation prediction](#32-observation-prediction)
   - [Gaussian update](#33-gaussian-update)
4. [RTS smoothing and the sufficient moments used by EM](#4-rts-smoothing-and-the-sufficient-moments-used-by-em)
5. [Generalized penalized EM](#5-generalized-penalized-em)
   - [Complete-data objective](#51-complete-data-objective)
   - [Transition update](#52-transition-update)
   - [Observation update as expected regression](#53-observation-update-as-expected-regression)
6. [Proposition 1: latent-coordinate invariance](#6-proposition-1-latent-coordinate-invariance)
   - [Full transformed parameter set](#61-full-transformed-parameter-set)
   - [State equation](#62-state-equation)
   - [Observation equation and induced operators](#63-observation-equation-and-induced-operators)
   - [Why fixing Q = I leaves an orthogonal ambiguity](#64-why-fixing-qi_d-leaves-an-orthogonal-ambiguity)
7. [Proposition 2: observable low-rank implication](#7-proposition-2-observable-low-rank-implication)
   - [Differencing and covariance](#71-differencing-and-covariance)
   - [Rank](#72-rank)
   - [Range](#73-range)
   - [Failure of full rank or excitation](#74-failure-of-full-rank-or-excitation)
8. [Sample low-rank and subspace diagnostics](#8-sample-low-rank-and-subspace-diagnostics)
   - [Increment covariance and SVD](#81-increment-covariance-and-svd)
   - [Principal angles and projector distance](#82-principal-angles-and-projector-distance)
   - [Perturbation interpretation](#83-perturbation-interpretation)
9. [Companion representation and coordinate responses](#9-companion-representation-and-coordinate-responses)
10. [Derivatives of propagation features](#10-derivatives-of-propagation-features)
    - [Derivative of a matrix power](#101-derivative-of-a-matrix-power)
    - [Response and energy derivatives](#102-response-and-energy-derivatives)
11. [Frozen gain, spectral stability, and non-normality](#11-frozen-gain-spectral-stability-and-non-normality)
12. [Proposition 3: exact frozen-versus-chronological decomposition](#12-proposition-3-exact-frozen-versus-chronological-decomposition)
13. [Corollary 1: affine state-path bound](#13-corollary-1-affine-state-path-bound)
14. [Migration-gap bound for frozen and pathwise gains](#14-migration-gap-bound-for-frozen-and-pathwise-gains)
15. [Conditional state-path uncertainty](#15-conditional-state-path-uncertainty)
16. [Downside target and predictive comparisons](#16-downside-target-and-predictive-comparisons)
    - [Five-day downside target](#161-five-day-downside-target)
    - [Ridge estimation and Validation selection](#162-ridge-estimation-and-validation-selection)
    - [Relative MSE improvement](#163-relative-mse-improvement)
    - [Paired moving-block percentile interval](#164-paired-moving-block-percentile-interval)
    - [Paired finite-seed Student interval](#165-paired-finite-seed-student-interval)
17. [Logical scope of the mathematical results](#17-logical-scope-of-the-mathematical-results)

## 1. Notation, conditioning, and timing

Let

- $x_t\in\mathbb R^N$ be the standardized sector-return vector;
- $u_t\in\mathbb R^q$ be the observed covariate vector;
- $z_t\in\mathbb R^d$ be the latent operator state;
- $A_\ell(z_t)\in\mathbb R^{N\times N}$ be the lag $\ell$ operator;
- $\eta_t\sim\mathcal N(0,Q)$ and $\varepsilon_t\sim\mathcal N(0,R)$ be mutually independent state and observation innovations.

The model is

$$
z_t=Fz_{t-1}+Gu_t+\eta_t,
\tag{1.1}
$$

$$
A_\ell(z_t)=A_\ell^{(0)}+\sum_{r=1}^d z_{r,t}B_{\ell,r},
\tag{1.2}
$$

and

$$
x_t=c+\sum_{\ell=1}^L A_\ell(z_t)x_{t-\ell}
     +\Lambda z_t+Du_t+\varepsilon_t.
\tag{1.3}
$$

The lagged returns in (1.3) are already observed when the conditional distribution at time $t$ is formed. Consequently, conditional on the past returns and the supplied covariate path, (1.3) is a linear Gaussian observation equation in $z_t$, even though the joint model contains products of a latent state and lagged returns.

For ex-ante prediction, write $\mathcal F_{t-1}$ for the information available immediately before recording the forecast for date $t$. The paper uses the deterministic persistence plug-in

$$
\widehat u_{t\mid t-1}=u_{t-1}.
\tag{1.4}
$$

The contemporaneous value $u_t$ and observation $x_t$ may update the posterior only after that forecast has been recorded.

## 2. Linearization of the observation equation

Insert (1.2) into (1.3):

$$
\begin{aligned}
x_t
&=c+Du_t+\sum_{\ell=1}^L A_\ell^{(0)}x_{t-\ell}
  +\sum_{\ell=1}^L\sum_{r=1}^d
    z_{r,t}B_{\ell,r}x_{t-\ell}
  +\Lambda z_t+\varepsilon_t\cr
&=c+Du_t+\sum_{\ell=1}^L A_\ell^{(0)}x_{t-\ell}
  +\sum_{r=1}^d z_{r,t}
    \left(\Lambda_{:,r}+\sum_{\ell=1}^L
    B_{\ell,r}x_{t-\ell}\right)
  +\varepsilon_t.
\end{aligned}
\tag{2.1}
$$

Define

$$
\mu_t=c+Du_t+\sum_{\ell=1}^L A_\ell^{(0)}x_{t-\ell}
\tag{2.2}
$$

and define the $r$-th column of $H_t\in\mathbb R^{N\times d}$ by

$$
H_{t,:,r}=\Lambda_{:,r}+\sum_{\ell=1}^L
B_{\ell,r}x_{t-\ell}.
\tag{2.3}
$$

Then

$$
x_t=\mu_t+H_tz_t+\varepsilon_t.
\tag{2.4}
$$

This algebra is the reason a standard time-varying-observation Kalman filter can be used: at each date, $H_t$ is known after conditioning on lagged returns and the current parameter values.

## 3. Kalman prediction, updating, and predictive covariance

### 3.1 State prediction

Suppose

$$
z_{t-1}\mid\mathcal F_{t-1}
\sim\mathcal N(m_{t-1},P_{t-1}).
\tag{3.1}
$$

Using (1.1), the one-step prior state is

$$
z_t\mid\mathcal F_{t-1}
=Fz_{t-1}+G\widehat u_{t\mid t-1}+\eta_t.
\tag{3.2}
$$

Linearity and independence of $z_{t-1}$ and $\eta_t$ give

$$
m_t^-=Fm_{t-1}+G\widehat u_{t\mid t-1},
\qquad
P_t^-=FP_{t-1}F^\top+Q.
\tag{3.3}
$$

### 3.2 Observation prediction

From (2.4),

$$
\widehat x_t
=\mathbb E[x_t\mid\mathcal F_{t-1}]
=\widehat\mu_t+H_tm_t^-.
\tag{3.4}
$$

Subtracting (3.4) from (2.4) gives the forecast error

$$
v_t=x_t-\widehat x_t
=H_t(z_t-m_t^-)+\varepsilon_t.
\tag{3.5}
$$

The two terms on the right are conditionally independent, so

$$
S_t
=\mathrm{Var}(v_t\mid\mathcal F_{t-1})
=H_tP_t^-H_t^\top+R.
\tag{3.6}
$$

This proves the predictive covariance formula stated in the paper. It includes filtered-state and process uncertainty through $P_t^-$, but, conditional on the fitted parameters, does not include parameter-estimation or support-selection uncertainty.

### 3.3 Gaussian update

The joint prior distribution of $z_t$ and $x_t$ is

$$
\begin{bmatrix}z_t\cr x_t\end{bmatrix}
\Bigm|\mathcal F_{t-1}
\sim\mathcal N\negthinspace\left(
\begin{bmatrix}m_t^-\cr\widehat\mu_t+H_tm_t^-\end{bmatrix},
\begin{bmatrix}
P_t^- & P_t^-H_t^\top\cr
H_tP_t^- & S_t
\end{bmatrix}
\right).
\tag{3.7}
$$

Conditioning a partitioned Gaussian distribution yields

$$
K_t=P_t^-H_t^\top S_t^{-1},
\qquad
m_t=m_t^-+K_tv_t,
\tag{3.8}
$$

and

$$
P_t=P_t^- - P_t^-H_t^\top S_t^{-1}H_tP_t^-.
\tag{3.9}
$$

For numerical stability, (3.9) can be evaluated in Joseph form,

$$
P_t=(I-K_tH_t)P_t^-(I-K_tH_t)^\top+K_tRK_t^\top,
\tag{3.10}
$$

which is algebraically equal to (3.9) when $K_t$ is given by (3.8).

The conditional Gaussian log likelihood contribution at date $t$ is

$$
\ell_t
=-\frac12\left[
N\log(2\pi)+\log|S_t|+v_t^\top S_t^{-1}v_t
\right].
\tag{3.11}
$$

Summing (3.11) over the training dates gives the observed-data likelihood used in fitting.

## 4. RTS smoothing and the sufficient moments used by EM

Filtering conditions on observations through date $t$; training-window smoothing conditions on the entire training sample. Let

$$
z_t\mid x_{1:T}\sim\mathcal N(m_t^s,P_t^s).
\tag{4.1}
$$

The one-step state prediction from date $t$ is

$$
m_{t+1}^-=Fm_t+Gu_{t+1},
\qquad
P_{t+1}^-=FP_tF^\top+Q.
\tag{4.2}
$$

The joint filtered/predicted Gaussian law implies the backward gain

$$
J_t=P_tF^\top(P_{t+1}^-)^{-1}.
\tag{4.3}
$$

Applying Gaussian conditioning to $(z_t,z_{t+1})$, then averaging over the smoothed distribution of $z_{t+1}$, gives the Rauch--Tung--Striebel recursions

$$
m_t^s=m_t+J_t(m_{t+1}^s-m_{t+1}^-),
\tag{4.4}
$$

$$
P_t^s=P_t+J_t(P_{t+1}^s-P_{t+1}^-)J_t^\top.
\tag{4.5}
$$

The required posterior moments are

$$
\mathbb E[z_tz_t^\top\mid x_{1:T}]
=P_t^s+m_t^s(m_t^s)^\top
\tag{4.6}
$$

and, using the lag-one smoothed covariance,

$$
\mathbb E[z_tz_{t-1}^\top\mid x_{1:T}]
=P_{t,t-1}^s+m_t^s(m_{t-1}^s)^\top,
\qquad
P_{t,t-1}^s=P_t^sJ_{t-1}^\top.
\tag{4.7}
$$

Smoothing is used only for fitting and training diagnostics. Ex-ante Test features use forward-filtered states, so future observations cannot enter the forecast-origin state estimate.

## 5. Generalized penalized EM

This section derives the objective and the main conditional updates. It describes the optimization target; it does not assert that the nonconvex procedure reaches a global optimum.

### 5.1 Complete-data objective

Let $\theta$ collect the observation parameters $(c,D,A^{(0)},\Lambda,B)$. Ignoring constants, the negative complete-data Gaussian log likelihood is

$$
\begin{aligned}
\mathcal L_c
=\frac12\sum_t
  &\left[
  \log|R|+(x_t-\mu_t-H_tz_t)^\top
  R^{-1}(x_t-\mu_t-H_tz_t)
  \right]\cr
+\frac12\sum_{t\ge2}
  &\left[
  \log|Q|+(z_t-Fz_{t-1}-Gu_t)^\top
  Q^{-1}(z_t-Fz_{t-1}-Gu_t)
  \right]\cr
+\frac12
  &\left[
  \log|P_0|+(z_1-m_0)^\top P_0^{-1}(z_1-m_0)
  \right].
\end{aligned}
\tag{5.1}
$$

The E-step replaces every latent first and second moment in (5.1) by (4.4)--(4.7). The penalized M-step minimizes the resulting expected objective plus

$$
\mathcal P(\theta)
=\lambda_g\sum_{\ell,j,i}w_{\ell ji}
  \sqrt{(A_{\ell,ji}^{(0)})^2+
        \sum_{r=1}^d B_{\ell,r,ji}^2}
+\lambda_B\sum_{\ell,r,j,i}w_{\ell ji}|B_{\ell,r,ji}|
+\frac{\lambda_2}{2}\Vert\theta\Vert_F^2.
\tag{5.2}
$$

The first term switches an entire baseline/state-dependent edge block together, the second sparsifies state-dependent loadings within active blocks, and the third stabilizes the remaining observation regression. Coefficients outside the screened support are constrained to zero.

### 5.2 Transition update

Define the augmented predictor

$$
\xi_t=\begin{bmatrix}z_{t-1}\cr u_t\end{bmatrix},
\qquad
W=[F\;G].
\tag{5.3}
$$

The transition residual is $z_t-W\xi_t$. With $Q=I_d$, minimizing its expected squared norm is equivalent to minimizing

$$
\sum_{t=2}^T
\mathbb E\left[
\Vert z_t-W\xi_t\Vert_2^2\mid x_{1:T}
\right].
\tag{5.4}
$$

Define

$$
S_{xx}=\sum_{t=2}^T\mathbb E[\xi_t\xi_t^\top\mid x_{1:T}],
\qquad
S_{yx}=\sum_{t=2}^T\mathbb E[z_t\xi_t^\top\mid x_{1:T}].
\tag{5.5}
$$

Expanding (5.4), differentiating with respect to $W$, and setting the gradient to zero gives

$$
-2S_{yx}+2WS_{xx}=0,
\qquad
W=S_{yx}S_{xx}^{\dagger},
\tag{5.6}
$$

where $\dagger$ denotes the Moore--Penrose inverse. Partitioning $W$ recovers $F$ and $G$. If $Q$ were unconstrained, its usual update would be

$$
Q=\frac{1}{T-1}
\sum_{t=2}^T
\mathbb E[(z_t-W\xi_t)(z_t-W\xi_t)^\top\mid x_{1:T}].
\tag{5.7}
$$

The fitted model instead fixes $Q=I_d$ after choosing the latent scale; the transition normal equation (5.6) is still the conditional minimizer for $F,G$ under that normalization.

### 5.3 Observation update as expected regression

Let

$$
s_t=\begin{bmatrix}1\cr x_{t-1}\cr\vdots\cr x_{t-L}\end{bmatrix},
\qquad
d_t=\begin{bmatrix}1\cr u_t\cr x_{t-1}\cr\vdots\cr x_{t-L}\end{bmatrix}.
\tag{5.8}
$$

The observation model can be written row-wise as a regression on the augmented random feature

$$
\phi_t(z_t)=
\begin{bmatrix}
d_t\cr z_t\otimes s_t
\end{bmatrix},
\qquad
x_t=\Theta\phi_t(z_t)+\varepsilon_t.
\tag{5.9}
$$

Because $s_t,d_t$ are observed,

$$
\mathbb E[z_t\otimes s_t]=(m_t^s\otimes s_t)
\tag{5.10}
$$

and

$$
\mathbb E[(z_t\otimes s_t)(z_t\otimes s_t)^\top]
=\mathbb E[z_tz_t^\top]\otimes(s_ts_t^\top).
\tag{5.11}
$$

Set

$$
\Gamma=\frac1{T_o}\sum_t
\mathbb E[\phi_t\phi_t^\top\mid x_{1:T}],
\qquad
M=\frac1{T_o}\sum_t x_t
\mathbb E[\phi_t\mid x_{1:T}]^\top.
\tag{5.12}
$$

With $R$ fixed, the gradient of the smooth expected observation objective is

$$
\nabla_\Theta
=R^{-1}(\Theta\Gamma-M)+\lambda_2\Theta.
\tag{5.13}
$$

A proximal-gradient step first moves along $-\nabla_\Theta$. For each active edge, the state-dependent entries receive scalar soft-thresholding for the $\ell_1$ term,

$$
\mathrm{soft}(b,\tau\lambda_B)
=\mathrm{sign}(b)(|b|-\tau\lambda_B)_+,
\tag{5.14}
$$

followed by group shrinkage of the concatenated baseline and state-dependent edge vector $g$:

$$
\mathrm{prox}_{\tau\lambda_g\Vert\cdot\Vert_2}(g)
=\left(1-\frac{\tau\lambda_g}{\Vert g\Vert_2}\right)_+g.
\tag{5.15}
$$

After updating $\Theta$, the expected residual covariance is

$$
R=\frac1{T_o}\sum_t
\mathbb E[(x_t-\Theta\phi_t)(x_t-\Theta\phi_t)^\top
\mid x_{1:T}],
\tag{5.16}
$$

with a positive-eigenvalue floor (and, when configured, a diagonal restriction). Because the proximal subproblem is solved only to finite precision, the implementation backtracks the whole proposal until both the old-posterior expected objective and the observed penalized likelihood are non-worsening. This is a generalized-EM monotonicity guard, not a proof of global convergence.

## 6. Proposition 1: latent-coordinate invariance

### 6.1 Full transformed parameter set

Let $T\in\mathbb R^{d\times d}$ be nonsingular and define

$$
z_t^{\ast}=Tz_t.
\tag{6.1}
$$

The complete transformed parameter set is

$$
\begin{aligned}
\mathcal B^{\ast}&=\mathcal BT^{-1},
&\Lambda^{\ast}&=\Lambda T^{-1},\cr
F^{\ast}&=TFT^{-1},
&G^{\ast}&=TG,\cr
Q^{\ast}&=TQT^\top,
&m_0^{\ast}&=Tm_0,\cr
P_0^{\ast}&=TP_0T^\top.
\end{aligned}
\tag{6.2}
$$

### 6.2 State equation

Multiplying the original state equation by $T$ gives

$$
\begin{aligned}
z_t^{\ast}
&=TFz_{t-1}+TGu_t+T\eta_t\cr
&=(TFT^{-1})z_{t-1}^{\ast}+(TG)u_t+\eta_t^{\ast},
\end{aligned}
\tag{6.3}
$$

where

$$
\eta_t^{\ast}=T\eta_t\sim\mathcal N(0,TQT^\top)
=\mathcal N(0,Q^{\ast}).
\tag{6.4}
$$

Thus the transformed state process has exactly the same form.

### 6.3 Observation equation and induced operators

The varying coefficient vector satisfies

$$
\mathcal B^{\ast}z_t^{\ast}
=\mathcal BT^{-1}Tz_t
=\mathcal Bz_t.
\tag{6.5}
$$

Similarly,

$$
\Lambda^{\ast}z_t^{\ast}=\Lambda T^{-1}Tz_t=\Lambda z_t.
\tag{6.6}
$$

Therefore every conditional mean of $x_t$, every lagged operator $A_\ell(z_t)$, and hence the full observation law are unchanged.

Finally, right multiplication by an invertible matrix does not alter a matrix's column space:

$$
\mathrm{Range}(\mathcal B^{\ast})
=\lbrace\mathcal BT^{-1}v:v\in\mathbb R^d\rbrace
=\lbrace\mathcal Bw:w\in\mathbb R^d\rbrace
=\mathrm{Range}(\mathcal B).
\tag{6.7}
$$

This proves Proposition 1.

### 6.4 Why fixing $Q=I_d$ leaves an orthogonal ambiguity

Suppose both the original and transformed representations satisfy $Q=Q^{\ast}=I_d$. From (6.2),

$$
I_d=T I_d T^\top=TT^\top.
\tag{6.8}
$$

Hence $T$ must be orthogonal. The scale is fixed, but rotations, reflections, and sign changes remain. Individual latent axes are therefore not identified economic factors; the induced operators, the column space of $\mathcal B$, and invariant propagation functionals are the meaningful targets.

## 7. Proposition 2: observable low-rank implication

The supported coefficient vector is

$$
a_t=a_0+\mathcal Bz_t,
\qquad
\mathcal B\in\mathbb R^{p\times d}.
\tag{7.1}
$$

### 7.1 Differencing and covariance

Because $a_0$ is time invariant,

$$
\Delta a_t=a_t-a_{t-1}
=\mathcal B(z_t-z_{t-1})
=\mathcal B\Delta z_t.
\tag{7.2}
$$

After centering, linearity of covariance gives

$$
\begin{aligned}
\Sigma_{\Delta a}
&=\mathbb E[(\Delta a_t-\mathbb E\Delta a_t)
             (\Delta a_t-\mathbb E\Delta a_t)^\top]\cr
&=\mathcal B
  \mathbb E[(\Delta z_t-\mathbb E\Delta z_t)
             (\Delta z_t-\mathbb E\Delta z_t)^\top]
  \mathcal B^\top\cr
&=\mathcal B\Sigma_{\Delta z}\mathcal B^\top.
\end{aligned}
\tag{7.3}
$$

### 7.2 Rank

Under full excitation, $\Sigma_{\Delta z}\succ0$, so its symmetric square root is invertible. Let

$$
V=\mathcal B\Sigma_{\Delta z}^{1/2}.
\tag{7.4}
$$

Then

$$
\Sigma_{\Delta a}=VV^\top.
\tag{7.5}
$$

Since right multiplication by the invertible $\Sigma_{\Delta z}^{1/2}$ preserves rank,

$$
\mathrm{rank}(V)=\mathrm{rank}(\mathcal B)=d.
\tag{7.6}
$$

For any matrix $V$,

$$
\mathrm{Null}(VV^\top)=\mathrm{Null}(V^\top),
\tag{7.7}
$$

because

$$
y^\top VV^\top y=\Vert V^\top y\Vert_2^2.
\tag{7.8}
$$

Applying rank-nullity to (7.7) gives

$$
\mathrm{rank}(\Sigma_{\Delta a})
=\mathrm{rank}(VV^\top)
=\mathrm{rank}(V)=d.
\tag{7.9}
$$

### 7.3 Range

Both $VV^\top$ and $V$ have rank $d$, and

$$
\mathrm{Range}(VV^\top)\subseteq\mathrm{Range}(V).
\tag{7.10}
$$

The inclusion is between spaces of the same dimension, so equality holds. Also, invertibility of $\Sigma_{\Delta z}^{1/2}$ gives

$$
\mathrm{Range}(V)=\mathrm{Range}(\mathcal B).
\tag{7.11}
$$

Combining (7.10)--(7.11),

$$
\mathrm{Range}(\Sigma_{\Delta a})
=\mathrm{Range}(\mathcal B).
\tag{7.12}
$$

Equations (7.3), (7.9), and (7.12) prove Proposition 2.

### 7.4 Failure of full rank or excitation

Without either assumption,

$$
\mathrm{rank}(\Sigma_{\Delta a})
\le
\min\lbrace\mathrm{rank}(\mathcal B),
       \mathrm{rank}(\Sigma_{\Delta z})\rbrace
\le d.
\tag{7.13}
$$

Equality and range recovery need not hold. This is why an estimated rank below $d$ can reflect weak excitation, collinear operator directions, or estimation error, rather than disproving the affine representation by itself.

## 8. Sample low-rank and subspace diagnostics

These formulas clarify the diagnostics used to evaluate the observable implication of Proposition 2. They are descriptive finite-sample quantities, not confidence statements.

### 8.1 Increment covariance and SVD

Given a coefficient path $a_1,\ldots,a_T$, form the centered increment matrix

$$
D=
\begin{bmatrix}
(\Delta a_2-\overline{\Delta a})^\top\cr
\vdots\cr
(\Delta a_T-\overline{\Delta a})^\top
\end{bmatrix}
\in\mathbb R^{(T-1)\times p}.
\tag{8.1}
$$

If

$$
D=U\mathrm{diag}(s_1,\ldots,s_r)V^\top,
\qquad s_1\ge\cdots\ge s_r\ge0,
\tag{8.2}
$$

then the sample covariance is

$$
\widehat\Sigma_{\Delta a}
=\frac{D^\top D}{T-2}
=V\mathrm{diag}\negthinspace\left(
\frac{s_1^2}{T-2},\ldots,\frac{s_r^2}{T-2}
\right)V^\top.
\tag{8.3}
$$

Thus the leading right singular vectors estimate the leading coefficient subspace. The variance share explained by the first $d$ directions is

$$
\mathrm{EVR}_d
=\frac{\sum_{j=1}^d s_j^2}{\sum_{j=1}^r s_j^2}.
\tag{8.4}
$$

With $\pi_j=s_j^2/\sum_k s_k^2$, the effective rank is the exponential spectral entropy

$$
r_{\mathrm{eff}}
=\exp\negthinspace\left(-\sum_{j:\pi_j\gt 0}\pi_j\log\pi_j\right).
\tag{8.5}
$$

It equals one when all variation lies in one direction and increases as the spectrum becomes diffuse. The population/sample eigengap at dimension $d$ is

$$
\widehat\delta_d
=\widehat\lambda_d-\widehat\lambda_{d+1}.
\tag{8.6}
$$

### 8.2 Principal angles and projector distance

Let $U,V\in\mathbb R^{p\times d}$ have orthonormal columns spanning two candidate subspaces. The singular values of $U^\top V$ are

$$
\sigma_j(U^\top V)=\cos\theta_j,
\qquad
0\le\theta_1\le\cdots\le\theta_d\le\frac\pi2.
\tag{8.7}
$$

The largest principal-angle sine is

$$
\sin\theta_{\max}
=\sqrt{1-\sigma_{\min}(U^\top V)^2}.
\tag{8.8}
$$

For equal-dimensional subspaces,

$$
\Vert UU^\top-VV^\top\Vert_2
=\sin\theta_{\max}.
\tag{8.9}
$$

This explains why a projector distance near one means that at least one direction is nearly orthogonal across windows, independently of basis signs or rotations.

If $D_{\mathrm{val}}$ is the centered Validation increment matrix and $U_{\mathrm{train}}$ is the Train basis, the unexplained Validation fraction is

$$
r_{\mathrm{res}}
=\frac{\Vert D_{\mathrm{val}}(I-U_{\mathrm{train}}
U_{\mathrm{train}}^\top)\Vert_F^2}
       {\Vert D_{\mathrm{val}}\Vert_F^2}.
\tag{8.10}
$$

### 8.3 Perturbation interpretation

Let $\widehat\Sigma=\Sigma+E$, and suppose the target population eigenspace is separated from the remaining spectrum by a positive gap $\delta$. A conservative Davis--Kahan form used for interpretation is

$$
\Vert\sin\Theta(\widehat U,U)\Vert_2
\le \min\left\lbrace1,\frac{2\Vert E\Vert_2}{\delta}\right\rbrace.
\tag{8.11}
$$

The bound shows why concentration alone is insufficient: a small covariance perturbation can still produce a large subspace error when the eigengap is small.

## 9. Companion representation and coordinate responses

Define the lag stack

$$
y_t=(x_t^\top,x_{t-1}^\top,\ldots,x_{t-L+1}^\top)^\top
\in\mathbb R^{NL}
\tag{9.1}
$$

and the selector

$$
J=[I_N,0,\ldots,0],
\qquad x_t=Jy_t.
\tag{9.2}
$$

Ignoring additive inputs when comparing two trajectories exposed to the same inputs, the stacked dynamics are

$$
y_t=C(z_t)y_{t-1},
\tag{9.3}
$$

where

$$
C(z)=
\begin{bmatrix}
A_1(z)&A_2(z)&\cdots&A_L(z)\cr
I_N&0&\cdots&0\cr
&\ddots&\ddots&\vdots\cr
0&&I_N&0
\end{bmatrix}.
\tag{9.4}
$$

Only the first block row depends on $z$, so (1.2) implies

$$
C(z)=C^{\mathrm{base}}+\sum_{r=1}^d z_rC_r,
\tag{9.5}
$$

where the first block row of $C_r$ is $[B_{1,r},\ldots,B_{L,r}]$ and all other blocks are zero.

Apply a unit coordinate perturbation $e_i$ to the current observed vector. Its stacked form is $J^\top e_i$. Under a frozen state $z$, after $h$ steps the stacked difference is

$$
\Delta y_{t+h}=C(z)^hJ^\top e_i.
\tag{9.6}
$$

Selecting target coordinate $j$ gives

$$
k_{i\to j}(h;z)
=e_j^\top JC(z)^hJ^\top e_i,
\tag{9.7}
$$

which is the coordinate-response formula in the paper. It is a conditional dynamical response: it holds the latent state and external inputs fixed across the perturbed and reference trajectories and does not identify a structural shock.

The discounted propagation energy is

$$
E_{i\to j}^{(H)}(z)
=\sum_{h=1}^H\delta^{h-1}k_{i\to j}(h;z)^2.
\tag{9.8}
$$

Squaring removes the response sign. Therefore energy rankings are reported together with signed kernels. The log directionality contrast

$$
D_{ij}^{(H)}(z)
=\log\frac{\epsilon_0+E_{i\to j}^{(H)}(z)}
              {\epsilon_0+E_{j\to i}^{(H)}(z)}
\tag{9.9}
$$

is antisymmetric:

$$
D_{ji}^{(H)}(z)=-D_{ij}^{(H)}(z).
\tag{9.10}
$$

## 10. Derivatives of propagation features

The M5 state-susceptibility features use derivatives with respect to the latent state. These derivatives follow from the product rule for matrix powers.

### 10.1 Derivative of a matrix power

Let $C(\alpha)=C+\alpha E$. Repeated use of the product rule gives

$$
\left.\frac{\mathrm d}{\mathrm d\alpha}C(\alpha)^h
\right|_{\alpha=0}
=\sum_{m=0}^{h-1}C^mEC^{h-1-m}.
\tag{10.1}
$$

This can also be proved by induction. For $h=1$, the derivative is $E$. If the formula holds for $h$, then

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm d\alpha}C(\alpha)^{h+1}
&=\frac{\mathrm d C(\alpha)^h}{\mathrm d\alpha}C(\alpha)
  +C(\alpha)^hE,
\end{aligned}
\tag{10.2}
$$

which appends the missing $m=h$ term.

Under (9.5), $\partial C/\partial z_r=C_r$, so

$$
\frac{\partial C(z)^h}{\partial z_r}
=\sum_{m=0}^{h-1}C(z)^mC_rC(z)^{h-1-m}.
\tag{10.3}
$$

### 10.2 Response and energy derivatives

From (9.7) and (10.3),

$$
\frac{\partial k_{i\to j}(h;z)}{\partial z_r}
=e_j^\top J
\left[\sum_{m=0}^{h-1}C(z)^mC_rC(z)^{h-1-m}\right]
J^\top e_i.
\tag{10.4}
$$

Differentiating (9.8) gives

$$
\frac{\partial E_{i\to j}^{(H)}(z)}{\partial z_r}
=2\sum_{h=1}^H\delta^{h-1}
k_{i\to j}(h;z)
\frac{\partial k_{i\to j}(h;z)}{\partial z_r}.
\tag{10.5}
$$

An elasticity-like normalized susceptibility is

$$
\chi_{ij,r}(z)
=\frac{\partial E_{i\to j}^{(H)}(z)/\partial z_r}
       {\epsilon_0+E_{i\to j}^{(H)}(z)}.
\tag{10.6}
$$

Aggregating $\chi_{ij,r}^2$ over ordered off-diagonal pairs and then over $r$ produces a basis-dependent local sensitivity feature. It is a fitted state-susceptibility diagnostic, not a causal derivative.

## 11. Frozen gain, spectral stability, and non-normality

For a frozen companion matrix $C_0$, the full finite-horizon gain is

$$
G_H^{\mathrm{fr}}(C_0)
=\max_{1\le h\le H}\Vert C_0^h\Vert_2.
\tag{11.1}
$$

The operator norm has the variational interpretation

$$
\Vert C_0^h\Vert_2
=\max_{v\ne0}\frac{\Vert C_0^hv\Vert_2}{\Vert v\Vert_2},
\tag{11.2}
$$

so (11.1) is the largest possible amplification of a unit stacked-state perturbation over the reported horizon. Restricting both input and output to observed coordinates gives

$$
G_{H,\mathrm{obs}}^{\mathrm{fr}}(C_0)
=\max_{1\le h\le H}\Vert JC_0^hJ^\top\Vert_2.
\tag{11.3}
$$

If $C_0$ is normal, $C_0^\top C_0=C_0C_0^\top$, then it is unitarily diagonalizable and

$$
\Vert C_0^h\Vert_2=\rho(C_0)^h.
\tag{11.4}
$$

For a non-normal matrix, eigenvectors need not be orthogonal, and $\Vert C_0^h\Vert_2$ can exceed one for finite $h$ even when $\rho(C_0)\lt 1$. Therefore the companion-radius gate is an asymptotic frozen stability diagnostic, not a guarantee against finite-horizon amplification.

## 12. Proposition 3: exact frozen-versus-chronological decomposition

Let

$$
\Phi_{t,h}=C_hC_{h-1}\cdots C_1,
\qquad C_0=C_t.
\tag{12.1}
$$

Define the hybrid products

$$
P_m=(C_h\cdots C_{m+1})C_0^m,
\qquad m=0,1,\ldots,h,
\tag{12.2}
$$

with empty products equal to the identity. The endpoints are

$$
P_0=C_h\cdots C_1=\Phi_{t,h},
\qquad
P_h=C_0^h.
\tag{12.3}
$$

For $m=1,\ldots,h$, factor the common left and right products:

$$
\begin{aligned}
P_{m-1}-P_m
&=(C_h\cdots C_{m+1})C_mC_0^{m-1}
  -(C_h\cdots C_{m+1})C_0C_0^{m-1}\cr
&=(C_h\cdots C_{m+1})(C_m-C_0)C_0^{m-1}.
\end{aligned}
\tag{12.4}
$$

Summing (12.4) telescopes:

$$
\begin{aligned}
\Phi_{t,h}-C_0^h
&=P_0-P_h\cr
&=\sum_{m=1}^h(P_{m-1}-P_m)\cr
&=\sum_{m=1}^h
(C_h\cdots C_{m+1})(C_m-C_0)C_0^{m-1}.
\end{aligned}
\tag{12.5}
$$

This proves Proposition 3. No commutativity and no low-dimensional parametrization are used. The order of every matrix product in (12.5) is therefore essential.

For example, when $h=3$,

$$
\begin{aligned}
C_3C_2C_1-C_0^3
={}&C_3C_2(C_1-C_0)
+C_3(C_2-C_0)C_0\cr
&+(C_3-C_0)C_0^2.
\end{aligned}
\tag{12.6}
$$

The three terms replace the chronological operators by the frozen operator one position at a time.

## 13. Corollary 1: affine state-path bound

Under the affine companion representation,

$$
C_m-C_0
=\sum_{r=1}^d(z_{r,t+m}-z_{r,t})C_r.
\tag{13.1}
$$

Let $\Delta z_m=z_{t+m}-z_t$. The triangle inequality gives

$$
\Vert C_m-C_0\Vert_2
\le\sum_{r=1}^d|\Delta z_{m,r}|\thinspace\Vert C_r\Vert_2.
\tag{13.2}
$$

Apply scalar Cauchy--Schwarz to the vectors $(|\Delta z_{m,1}|,\ldots,|\Delta z_{m,d}|)$ and $(\Vert C_1\Vert_2,\ldots,\Vert C_d\Vert_2)$:

$$
\Vert C_m-C_0\Vert_2
\le
\left(\sum_{r=1}^d\Delta z_{m,r}^2\right)^{1/2}
\left(\sum_{r=1}^d\Vert C_r\Vert_2^2\right)^{1/2}.
\tag{13.3}
$$

With

$$
L_C=\left(\sum_{r=1}^d\Vert C_r\Vert_2^2\right)^{1/2},
\tag{13.4}
$$

equation (13.3) becomes

$$
\Vert C_m-C_0\Vert_2\le L_C\Vert z_{t+m}-z_t\Vert_2.
\tag{13.5}
$$

Assume $\Vert C_j\Vert_2\le\kappa$ for $j=0,\ldots,h$. The $m$-th term of (12.5) has $h-m$ matrices to the left of $C_m-C_0$ and $m-1$ frozen matrices to the right. Submultiplicativity therefore yields

$$
\begin{aligned}
&\Vert(C_h\cdots C_{m+1})(C_m-C_0)C_0^{m-1}\Vert_2\cr
&\quad\le
\kappa^{h-m}\Vert C_m-C_0\Vert_2\kappa^{m-1}\cr
&\quad\le
\kappa^{h-1}L_C\Vert z_{t+m}-z_t\Vert_2.
\end{aligned}
\tag{13.6}
$$

Finally, apply the triangle inequality to (12.5):

$$
\boxed{
\Vert\Phi_{t,h}-C_0^h\Vert_2
\le
\kappa^{h-1}L_C
\sum_{m=1}^h\Vert z_{t+m}-z_t\Vert_2
}.
\tag{13.7}
$$

This proves Corollary 1. It is a finite-horizon local bound. It can be loose, especially because it replaces all step-specific norms and cancellations by a single worst-case $\kappa$.

## 14. Migration-gap bound for frozen and pathwise gains

Define

$$
g_h=\Vert\Phi_{t,h}\Vert_2,
\qquad
f_h=\Vert C_0^h\Vert_2.
\tag{14.1}
$$

The reverse triangle inequality for a norm gives

$$
|g_h-f_h|
\le\Vert\Phi_{t,h}-C_0^h\Vert_2.
\tag{14.2}
$$

For arbitrary finite sequences $\lbrace g_h\rbrace$ and $\lbrace f_h\rbrace$,

$$
\left|\max_hg_h-\max_hf_h\right|
\le\max_h|g_h-f_h|.
\tag{14.3}
$$

To verify (14.3), let $h_g$ maximize $g_h$. Then

$$
\max_hg_h-\max_hf_h
\le g_{h_g}-f_{h_g}
\le\max_h|g_h-f_h|,
\tag{14.4}
$$

and exchanging $f$ and $g$ gives the opposite inequality.

Define the migration gap

$$
M_{H,t}
=\max_{1\le h\le H}\Vert\Phi_{t,h}-C_0^h\Vert_2.
\tag{14.5}
$$

Combining (14.2)--(14.5) yields

$$
\boxed{
\left|
G_{H,t}^{\mathrm{path}}-G_H^{\mathrm{fr}}(C_0)
\right|
\le M_{H,t}
}.
\tag{14.6}
$$

The bound concerns only approximation error from freezing the fitted operator path. It does not test low dimensionality, measure predictive value, or assign a causal interpretation.

## 15. Conditional state-path uncertainty

At a forecast origin, suppose

$$
z_t\mid\mathcal F_t\sim\mathcal N(m_t,P_t).
\tag{15.1}
$$

For a supplied future covariate path $u_{t+1:t+H}$, recursively simulate

$$
z_{t+h}^{(b)}
=Fz_{t+h-1}^{(b)}+Gu_{t+h}+\eta_{t+h}^{(b)},
\qquad
\eta_{t+h}^{(b)}\sim\mathcal N(0,Q).
\tag{15.2}
$$

The analytic moments satisfy

$$
m_{t+h\mid t}=Fm_{t+h-1\mid t}+Gu_{t+h},
\tag{15.3}
$$

$$
P_{t+h\mid t}=FP_{t+h-1\mid t}F^\top+Q.
\tag{15.4}
$$

For each draw, construct $C(z_{t+1}^{(b)}),\ldots,C(z_{t+H}^{(b)}))$ and evaluate the pathwise gain or migration gap. Means, sample standard deviations, and equal-tailed quantiles over draws estimate the corresponding conditional functionals. These draws propagate filtered-state and future-process uncertainty while holding parameters, support, and the future-covariate scenario fixed.

For one affine edge coefficient

$$
a(z_t)=a_0+b^\top z_t,
\tag{15.5}
$$

Gaussian closure gives

$$
a(z_t)\mid\mathcal F_t
\sim\mathcal N(a_0+b^\top m_t,b^\top P_tb).
\tag{15.6}
$$

Therefore, when $b^\top P_tb\gt 0$,

$$
\Pr(a(z_t)\gt 0\mid\mathcal F_t)
=\Phi\negthinspace\left(
\frac{a_0+b^\top m_t}{\sqrt{b^\top P_tb}}
\right).
\tag{15.7}
$$

This is conditional sign uncertainty, not uncertainty over the fitted edge support or parameter estimates.

## 16. Downside target and predictive comparisons

### 16.1 Five-day downside target

For raw market simple returns $r_{t+h}^{\mathrm{mkt}}$, the target is

$$
Y_{t,5}=\frac15\sum_{h=1}^5
\left[\min\lbrace r_{t+h}^{\mathrm{mkt}},0\rbrace\right]^2.
\tag{16.1}
$$

Equivalently, with $r^- =\min(r,0)$,

$$
Y_{t,5}=\frac15\sum_{h=1}^5(r_{t+h}^-)^2.
\tag{16.2}
$$

The fixed denominator means a positive-return day contributes zero rather than being removed. Moreover,

$$
\frac15\sum_{h=1}^5(r_{t+h}^-)^2
\ne
\left(\min\left\lbrace\sum_{h=1}^5r_{t+h},0\right\rbrace\right)^2
\tag{16.3}
$$

in general, so the target is an average of daily downside squares, not the downside of the cumulative five-day return.

### 16.2 Ridge estimation and Validation selection

For model $M_k$, Train estimation solves

$$
(\widehat\beta_{k,\alpha},\widehat b_{k,\alpha})
=\arg\min_{\beta,b}
\sum_{t\in\mathrm{Train}}(Y_{t,5}-b-X_{k,t}^\top\beta)^2
+\alpha\Vert\beta\Vert_2^2,
\tag{16.4}
$$

where the intercept is not penalized. Validation chooses

$$
\widehat\alpha_k
=\arg\min_{\alpha\in\lbrace0.01,0.1,1,10,100\rbrace}
\frac1{|\mathrm{Val}|}
\sum_{t\in\mathrm{Val}}
(Y_{t,5}-\widehat Y_{k,t}^{(\alpha)})^2.
\tag{16.5}
$$

The selected Train fit is carried unchanged to Test; there is no post-selection Train-plus-Validation refit. Predictions are clipped at zero because the target is nonnegative.

### 16.3 Relative MSE improvement

Let $\ell_{A,t}$ and $\ell_{B,t}$ be squared Test losses for a reference model $A$ and candidate $B$. Their Test MSEs are

$$
\overline\ell_A=\frac1n\sum_t\ell_{A,t},
\qquad
\overline\ell_B=\frac1n\sum_t\ell_{B,t}.
\tag{16.6}
$$

The reported relative improvement is

$$
R_{A\to B}=1-\frac{\overline\ell_B}{\overline\ell_A}.
\tag{16.7}
$$

Thus $R_{A\to B}\gt 0$ means that the candidate has lower MSE, while a negative value means that adding the candidate feature block increases MSE.

### 16.4 Paired moving-block percentile interval

Let $L$ be the block length. A bootstrap draw samples starting indices and concatenates overlapping consecutive Test blocks until $n$ indices have been collected. Crucially, the same index vector $I_b$ is applied to both models. For draw $b$, compute

$$
R_{A\to B}^{\ast(b)}
=1-
\frac{|I_b|^{-1}\sum_{t\in I_b}\ell_{B,t}}
     {|I_b|^{-1}\sum_{t\in I_b}\ell_{A,t}}.
\tag{16.8}
$$

The pointwise $100(1-\gamma)$% percentile interval is

$$
\left[
q_{\gamma/2}\lbrace R_{A\to B}^{\ast(b)}\rbrace,
q_{1-\gamma/2}\lbrace R_{A\to B}^{\ast(b)}\rbrace
\right].
\tag{16.9}
$$

Pairing preserves contemporaneous covariance between the two loss sequences; blocks preserve local serial dependence. The interval conditions on the already selected models, supports, penalties, features, and forecasts. It is not a full-procedure bootstrap.

### 16.5 Paired finite-seed Student interval

For a synthetic metric, let $d_s=m_{\mathrm{DLC},s}-m_{\mathrm{comp},s}$ be the matched difference for seed $s=1,\ldots,S$. Define

$$
\bar d=\frac1S\sum_{s=1}^Sd_s,
\qquad
s_d^2=\frac1{S-1}\sum_{s=1}^S(d_s-\bar d)^2.
\tag{16.10}
$$

The descriptive two-sided $100(1-\gamma)$% interval is

$$
\bar d\ \pm\ t_{S-1,1-\gamma/2}\frac{s_d}{\sqrt S}.
\tag{16.11}
$$

Because the seeds are matched, (16.11) uses variation in the within-seed contrast rather than treating the two methods as independent samples. With a small fixed seed set, it remains a descriptive finite-replication summary and is not a multiplicity-adjusted confirmatory test.

## 17. Logical scope of the mathematical results

The derivations imply the following distinctions.

1. Proposition 1 identifies the invariant target: fitted operators and their shared column space, not named latent axes.
2. Proposition 2 gives a population implication of an exact affine low-dimensional coefficient representation under full rank and excitation. A low-rank path produced by the restricted model itself cannot independently validate that implication.
3. Proposition 3 is an algebraic identity for any conformable time-varying operator sequence. It does not require low-dimensional variation.
4. Corollary 1 adds an affine state-path upper bound to Proposition 3. It is local, finite-horizon, and conditional on a uniform operator-norm bound.
5. The migration gap measures frozen-operator approximation error. Neither it nor transient gain establishes compression, predictive usefulness, causal impact, or downside risk.
6. State-path draws, finite-seed intervals, and moving-block intervals address different sources of variation. None propagates the complete uncertainty of support selection, model selection, and parameter estimation.
