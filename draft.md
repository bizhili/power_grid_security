### Problem with answer
The problem is:
I have two equations: $(\eta+\epsilon)^T(\eta+\epsilon)=1$, $Z^T\epsilon+E^T\eta+E^T\epsilon=0$, where $Z, E$ are $n \times \tau$ matrices, $\epsilon$, $\eta$ are $n\times 1$ vectors. I want to solve $\epsilon$ using $Z, E$, and $\eta$ using the Lagrange multiplier method.
$\eta$ is unit vector such that $\eta^T \eta$= 1.
$\eta$ belongs to the null space of $Z^T$ such that: $Z^T \eta=0$.
E is zero mean Gaussian  noise matrix with std $\sigma$
$Z=HX$, $H\in R^{n\times m}$ is the system matrix. $X\in R^{m\times \tau}$ is the Gaussian random state with zero mean and std=1.

$$\min_{\epsilon} \frac{1}{2} \|Z\epsilon + E^T \eta + E^T \epsilon\|^2 \quad \text{s.t.} \quad |\eta+\epsilon|^2 = 1$$

The exact compact solution:

$$\epsilon= [I_m+K+L+M-\lambda G]^{-1}[-(K+M)\eta+\alpha \eta]$$

Where: 
$$K=(Z^+)^TE,L=(ZZ^T)^+EZ^T,M=(ZZ^T)EE^T,G=(ZZ^T)^+$$

$$\lambda=\|\bar{Z}(\eta+\epsilon)\|^2$$

$$\alpha = \frac{w^Tv - \eta^Tw \pm \sqrt{(\eta^Tw - w^Tv)^2 - \|w\|^2(\|v\|^2 - 2\eta^Tv)}}{\|w\|^2}$$

where:
- $w = (I_m + K + L + M - \lambda G)^{-1}\eta$
- $v = (I_m + K + L + M - \lambda G)^{-1}(K+M)\eta$


## Small  noise 

Up to order $K$:

Initialise:

$\epsilon_0= \eta, \epsilon_{-1}=0, \lambda_0=0$.

Let: 

$\mathcal{E}=E/\sigma,G= (ZZ^T)^+, H_1= Z\mathcal{E}^T+\mathcal{E}Z^T, H_2=\mathcal{E}\mathcal{E}^T$

For $k=1,2,3...$:

Eigenvalue:

$$\lambda_k=\eta^TH_1\epsilon_{k-1}+\eta^TH_2\epsilon_{k-2}-\sum_{j=1}^{k-1}\lambda_j(\eta^T\epsilon_{k-j})$$

Perpendicular component:

$$\epsilon_{k}^{\perp}=G(-H_1\epsilon_{k-1}-H_2\epsilon_{k-2}+\sum_{j=1}^{k-1}\lambda_j\epsilon_{k-j})$$

Parallel component:

$$c_k=-\frac{1}{2}\sum_{j=1}^{k-1}\epsilon_j^T\epsilon_{k-j}$$

Combine:

$$\epsilon_k=\epsilon_{k}^{\perp}+c_k \eta$$

Finally:

$$\epsilon=\sum_{k=1}^{K}\sigma^{k}\epsilon_k;\lambda=\sum_{k=1}^{K}\sigma^{k}\lambda_k$$

Zeroth order, K=0:

$$\epsilon=0+O(\sigma)$$


First order, K=1:

$$\epsilon=-\sigma(Z)^+Z\mathcal{E}\eta+O(\sigma^2)$$

Second order, K=2:

$$\epsilon=-\sigma(Z)^+Z\mathcal{E}\eta-\sigma^2G(H_2\eta-\sigma H_1(Z)^+Z\mathcal{E}\eta)-\sigma^2\frac{1}{2}|\sigma(Z)^+Z\mathcal{E}\eta|^2\eta+O(\sigma^3)$$


## Large noise

Up to order $K$:

Solve the eigenvalue problem $\mathcal{E}\mathcal{E}^T\bar{\eta}=\bar{\lambda}_0\bar{\eta}$ for the smallest eigenvalue/eigenvector pair.

Initialise:

$\bar{\epsilon}_0=\bar{\eta},\bar{\epsilon}_{-1}=0$

Let $R_0=(\mathcal{E}\mathcal{E}^T-\bar{\lambda}_0I)^+$

Eigenvalue:
$$\bar{\lambda}_k=\bar{\eta}H_1\bar{\epsilon}_{k-1}+\bar{\eta}H_2\bar{\epsilon}_{k-2}-\sum_{j=1}^{k-1}\bar{\lambda}_{j}(\bar{\eta}^T\bar{\epsilon}_{k-j})$$

Perpendicular compone:


$$\bar{\epsilon}_{k}^{\perp}=R_0(-H_1\bar{\epsilon}_{k-1}-H_2\bar{\epsilon}_{k-2}+\sum_{j=1}^{k-1}\bar{\lambda}_j\bar{\epsilon}_{k-j})$$


Parallel component:

$$\bar{c}_k=-\frac{1}{2}\sum_{j=1}^{k-1}\bar{\epsilon}_j^T\bar{\epsilon}_{k-j}$$

Combine:

$$\bar{\epsilon}_k=\bar{\epsilon}_{k}^{\perp}+\bar{c}_k\bar{\eta}$$

Finally:

$$\bar{\epsilon}=\bar{\eta}+\sum_{k=1}^K\frac{\bar{\epsilon}_k}{\sigma^k}; \bar{\lambda}= \bar{\lambda}_0+\sum_{k=1}^{K}\frac{\bar{\lambda}_k}{\sigma^k}$$

Zeroth order, K=0:

$$\bar{\epsilon}=\bar{\eta}-\eta+O(1/\sigma)$$


First order, K=1:

$$\bar{\epsilon}=-(\mathcal{E}\mathcal{E}^T-\bar{\lambda}_0I)^+(Z\mathcal{E}^T+\mathcal{E}Z^T)\bar{\eta}/\sigma+\bar{\eta}-\eta+O(1/\sigma^2)$$