"""
Dixon-Coles (1997) low-score correlation correction, tau(x, y; lam, mu, rho).

    tau(0,0) = 1 - lam*mu*rho
    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1                otherwise

The joint scoreline probability is P(x,y) = tau(x,y) * Poisson(x; lam) *
Poisson(y; mu). This module holds a single scalar reference implementation
that both the PyMC likelihood (dixon_coles.py, via static masks + pytensor
switch, for speed) and the posterior-predictive scoreline grid
(predict.py, via numpy) are checked against in tests — the vectorized
versions in those two files must agree with this function for every
(x, y, lam, mu, rho) combination.
"""


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0
