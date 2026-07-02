# Model Methodology

## 1. GJR-GARCH(1,1) Variance Process

The conditional variance at time $t$ is given by:

$$\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \lambda_g \varepsilon^2_{t-1} \mathbf{1}[\varepsilon_{t-1} < 0] + \beta \sigma^2_{t-1}$$

Where:

- $\omega$ = unconditional variance floor
- $\alpha$ = symmetric shock response
- $\lambda_g$ = asymmetric leverage effect — negative shocks increase variance more than positive shocks of equal magnitude
- $\beta$ = variance persistence
- $\mathbf{1}[\varepsilon_{t-1} < 0]$ = indicator function, 1 if previous return was negative

Stationarity requires:

$$\alpha + \frac{\lambda_g}{2} + \beta < 1$$

---

## 2. Two-State Hidden Markov Model

The market is assumed to occupy one of two latent states at each time $t$:

- **State 0 (Bull):** low-variance regime, $\sigma^2_{t,0} = \sigma^2_t$
- **State 1 (Bear):** high-variance regime, $\sigma^2_{t,1} = \gamma \cdot \sigma^2_t$

Where $\gamma > 1$ is the bear-state variance multiplier. Returns are assumed normally distributed conditional on state:

$$r_t \mid S_t = 0 \sim \mathcal{N}(0,\ \sigma^2_t)$$

$$r_t \mid S_t = 1 \sim \mathcal{N}(0,\ \gamma \cdot \sigma^2_t)$$

---

## 3. Hamilton Filter (Bayesian State Updating)

At each time $t$, the filter updates state probabilities via Bayes' theorem:

$$P(S_t = j \mid r_1, \ldots, r_t) \propto P(r_t \mid S_t = j) \cdot P(S_t = j \mid r_1, \ldots, r_{t-1})$$

The prediction step propagates probabilities forward using the transition matrix:

$$P(S_t = 0 \mid r_{t-1}) = P(S_{t-1} = 0) \cdot p_{00} + P(S_{t-1} = 1) \cdot p_{10}$$

$$P(S_t = 1 \mid r_{t-1}) = P(S_{t-1} = 0) \cdot p_{01} + P(S_{t-1} = 1) \cdot p_{11}$$

Initialization uses the ergodic (steady-state) probability derived from the transition matrix at $\text{VIX}_0$:

$$P(S_0 = 0) = \frac{1 - p_{11}}{(1 - p_{00}) + (1 - p_{11})}$$

---

## 4. Time-Varying Transition Probabilities (TVTP)

Transition probabilities are driven by lagged VIX through a logistic function:

$$p_{01}(t) = P(S_t = 1 \mid S_{t-1} = 0) = \frac{1}{1 + \exp\left(-(b_{0,01} + b_{1,01} \cdot \text{VIX}_{t-1})\right)}$$

$$p_{10}(t) = P(S_t = 0 \mid S_{t-1} = 1) = \frac{1}{1 + \exp\left(-(b_{0,10} + b_{1,10} \cdot \text{VIX}_{t-1})\right)}$$

When VIX is elevated, $p_{01}$ rises (higher probability of transitioning to bear).  
When VIX normalizes, $p_{10}$ rises (higher probability of returning to bull).

---

## 5. Maximum Likelihood Estimation

Parameters $\theta = [\omega,\ \alpha,\ \beta,\ \lambda_g,\ \gamma,\ b_{0,01},\ b_{1,01},\ b_{0,10},\ b_{1,10}]$ are estimated by maximizing the log-likelihood of observed returns:

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \log \left[ P(S_t = 0) \cdot f(r_t \mid S_t = 0) + P(S_t = 1) \cdot f(r_t \mid S_t = 1) \right]$$

Where $f(r_t \mid S_t = j)$ is the Gaussian density evaluated at state $j$'s conditional variance.

Optimization uses L-BFGS-B with random restarts to avoid local optima. The stationarity constraint is enforced as a soft penalty in the objective function.

---

## 6. Trading Rule

Let $\hat{\pi}_t = P(S_t = 1 \mid r_1, \ldots, r_t)$ be the filtered bear-state probability.

**Smoothing:** apply an exponentially weighted moving average with span 20:

$$\tilde{\pi}_t = \text{EWM}(\hat{\pi}_t,\ \text{span}=20)$$

**Slope signal:** 20-day finite difference:

$$\Delta_t = \tilde{\pi}_t - \tilde{\pi}_{t-20}$$

**Trading rule** (signal shifted +1 day to prevent look-ahead bias):

$$\text{Signal}_{t} = \begin{cases} \text{XLU} & \text{if } \Delta_{t-1} > 0 \quad \text{(fear rising)} \\ \text{QQQ} & \text{if } \Delta_{t-1} \leq 0 \quad \text{(fear flat or falling)} \end{cases}$$

---

## References

- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series. _Econometrica_, 57(2), 357–384.
- Hamilton, J.D. & Susmel, R. (1994). Autoregressive conditional heteroskedasticity and changes in regime. _Journal of Econometrics_, 64, 307–333.
- Filardo, A.J. (1994). Business-cycle phases and their transitional dynamics. _Journal of Business & Economic Statistics_, 12(3), 299–308.
- Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. _Journal of Finance_, 48(5), 1779–1801.
