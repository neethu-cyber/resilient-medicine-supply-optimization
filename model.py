"""Two-stage stochastic mixed-integer program for a resilient medicine supply network.

First stage (decided now, before demand and disruptions are known):
    y[w] in {0,1}  open warehouse w
    q[w] >= 0      packs pre-positioned at warehouse w

Second stage (decided per scenario k, after uncertainty is revealed):
    x[w,h,k] >= 0  packs shipped from warehouse w to hospital h
    s[h,k]   >= 0  unmet demand (shortage) at hospital h

min  sum_w f[w] y[w] + hold * sum_w q[w]
     + sum_k p[k] * ( sum_{w,h} c[w,h] x[w,h,k] + penalty * sum_h s[h,k] )

s.t. q[w] <= cap[w] * y[w]                          (stock only at open warehouses)
     sum_w x[w,h,k] + s[h,k] = d[h,k]               (every hospital's demand is served or counted as shortage)
     sum_h x[w,h,k] <= a[w,k] * q[w]                (ship only stock that survives a disruption)
     s[h,k] <= (1 - alpha) * d[h,k]   (optional)    (fairness: minimum service level alpha for EVERY hospital)

With a single expected-value scenario this reduces to the deterministic capacitated
facility location problem.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


class MILPBuilder:
    """Small helper that collects variables and constraints, then calls SciPy's HiGHS MILP solver."""

    def __init__(self):
        self.cost, self.lb, self.ub, self.integer = [], [], [], []
        self.rows = []  # list of (dict var_index -> coefficient, lower, upper)

    def add_var(self, cost=0.0, lb=0.0, ub=np.inf, integer=False):
        self.cost.append(cost)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integer.append(1 if integer else 0)
        return len(self.cost) - 1

    def add_constr(self, coefs: dict, lb=-np.inf, ub=np.inf):
        self.rows.append((coefs, lb, ub))

    def solve(self, time_limit=120, mip_gap=1e-4):
        n, m = len(self.cost), len(self.rows)
        A = lil_matrix((m, n))
        lo, hi = np.empty(m), np.empty(m)
        for i, (coefs, l, u) in enumerate(self.rows):
            for j, v in coefs.items():
                A[i, j] = v
            lo[i], hi[i] = l, u
        res = milp(
            c=np.array(self.cost),
            constraints=LinearConstraint(A.tocsr(), lo, hi),
            integrality=np.array(self.integer),
            bounds=Bounds(np.array(self.lb), np.array(self.ub)),
            options={"time_limit": time_limit, "mip_rel_gap": mip_gap},
        )
        if res.status not in (0, 1) or res.x is None:
            raise RuntimeError(f"Solver failed: {res.message}")
        return res


@dataclass
class Solution:
    objective: float
    open: np.ndarray        # (W,) 0/1
    stock: np.ndarray       # (W,)
    flows: np.ndarray       # (W, H, K)
    shortage: np.ndarray    # (H, K)
    fixed_cost: float
    holding_cost: float
    transport_cost: float   # expected
    shortage_cost: float    # expected


def solve_network(inst, scenarios, min_service=None, fixed_open=None, fixed_stock=None, robust=False):
    """Build and solve the two-stage model.

    min_service : e.g. 0.9 forces every hospital to receive >= 90% of its demand in every scenario.
    fixed_open / fixed_stock : fix the first-stage decisions (used to evaluate a given plan).
    robust : if True, minimise the WORST-CASE second-stage cost over all scenarios (min-max robust
             optimization) instead of the expected cost:
                 min  first-stage cost + t     s.t.  t >= second-stage cost of scenario k   for all k
    """
    W, H, K = inst.W, inst.H, len(scenarios)
    m = MILPBuilder()

    y = []
    q = []
    for w in range(W):
        if fixed_open is not None:
            v = float(round(fixed_open[w]))
            y.append(m.add_var(inst.fixed_cost[w], v, v, integer=True))
        else:
            y.append(m.add_var(inst.fixed_cost[w], 0, 1, integer=True))
        if fixed_stock is not None:
            q.append(m.add_var(inst.holding_cost, fixed_stock[w], fixed_stock[w]))
        else:
            q.append(m.add_var(inst.holding_cost))
        m.add_constr({q[w]: 1.0, y[w]: -inst.capacity[w]}, ub=0.0)

    x = np.empty((W, H, K), dtype=int)
    s = np.empty((H, K), dtype=int)
    t = m.add_var(1.0, lb=-np.inf) if robust else None
    for k, sc in enumerate(scenarios):
        weight = 0.0 if robust else sc.prob   # robust: scenario costs enter through t instead
        scen_cost = {}
        for w in range(W):
            for h in range(H):
                x[w, h, k] = m.add_var(weight * inst.transport_cost[w, h])
                scen_cost[int(x[w, h, k])] = inst.transport_cost[w, h]
        for h in range(H):
            s[h, k] = m.add_var(weight * inst.shortage_penalty)
            scen_cost[int(s[h, k])] = inst.shortage_penalty
            coefs = {int(x[w, h, k]): 1.0 for w in range(W)}
            coefs[int(s[h, k])] = 1.0
            m.add_constr(coefs, lb=sc.demand[h], ub=sc.demand[h])
            if min_service is not None:
                m.add_constr({int(s[h, k]): 1.0}, ub=(1 - min_service) * sc.demand[h])
        for w in range(W):
            coefs = {int(x[w, h, k]): 1.0 for h in range(H)}
            coefs[q[w]] = -sc.availability[w]
            m.add_constr(coefs, ub=0.0)
        if robust:
            m.add_constr({t: 1.0, **{j: -c for j, c in scen_cost.items()}}, lb=0.0)

    res = m.solve()
    v = res.x
    open_ = np.round(v[y])
    stock = v[q]
    flows = v[x]
    short = v[s]
    probs = np.array([sc.prob for sc in scenarios])
    transport = float(np.sum(flows * inst.transport_cost[:, :, None] * probs[None, None, :]))
    shortage = float(np.sum(short * probs[None, :]) * inst.shortage_penalty)
    return Solution(
        objective=float(res.fun),
        open=open_,
        stock=stock,
        flows=flows,
        shortage=short,
        fixed_cost=float(open_ @ inst.fixed_cost),
        holding_cost=float(stock.sum() * inst.holding_cost),
        transport_cost=transport,
        shortage_cost=shortage,
    )
