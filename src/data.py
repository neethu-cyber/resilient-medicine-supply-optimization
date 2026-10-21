"""Synthetic but realistic data for a German medicine distribution network.

Candidate warehouses and hospitals are placed at real city coordinates.
Distances are great-circle (haversine) distances in km.
All numbers are weekly figures for one critical medicine (units = packs).
"""
from dataclasses import dataclass, field

import numpy as np

# (name, latitude, longitude, weekly fixed cost to operate [EUR], capacity [packs])
WAREHOUSES = [
    ("Nuernberg", 49.45, 11.08, 3200, 1400),
    ("Frankfurt", 50.11, 8.68, 3600, 1600),
    ("Muenchen", 48.14, 11.58, 3800, 1600),
    ("Leipzig", 51.34, 12.37, 2800, 1200),
    ("Stuttgart", 48.78, 9.18, 3300, 1300),
    ("Hannover", 52.37, 9.74, 3000, 1300),
]

# (name, latitude, longitude, expected weekly demand [packs])
HOSPITALS = [
    ("Berlin", 52.52, 13.40, 320),
    ("Hamburg", 53.55, 9.99, 260),
    ("Koeln", 50.94, 6.96, 240),
    ("Dresden", 51.05, 13.74, 150),
    ("Regensburg", 49.01, 12.10, 110),
    ("Wuerzburg", 49.79, 9.95, 100),
    ("Augsburg", 48.37, 10.90, 130),
    ("Mannheim", 49.49, 8.47, 140),
    ("Kassel", 51.31, 9.48, 90),
    ("Erfurt", 50.98, 11.03, 100),
    ("Bremen", 53.08, 8.80, 150),
    ("Freiburg", 47.99, 7.85, 120),
    ("Bamberg", 49.89, 10.89, 70),
    ("Dortmund", 51.51, 7.47, 200),
    ("Passau", 48.57, 13.43, 60),
]

TRANSPORT_COST_PER_KM = 0.012  # EUR per pack per km
HOLDING_COST = 1.5            # EUR per pack pre-positioned per week
SHORTAGE_PENALTY = 60.0       # EUR per pack of unmet hospital demand


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass
class Instance:
    warehouses: list
    hospitals: list
    fixed_cost: np.ndarray
    capacity: np.ndarray
    base_demand: np.ndarray
    transport_cost: np.ndarray          # shape (W, H)
    holding_cost: float = HOLDING_COST
    shortage_penalty: float = SHORTAGE_PENALTY
    distance_km: np.ndarray = field(default=None)

    @property
    def W(self):
        return len(self.warehouses)

    @property
    def H(self):
        return len(self.hospitals)


@dataclass
class Scenario:
    prob: float
    demand: np.ndarray        # shape (H,)
    availability: np.ndarray  # shape (W,), share of warehouse stock usable (1 = fully available)
    label: str = ""


def build_instance() -> Instance:
    w_lat = np.array([w[1] for w in WAREHOUSES])
    w_lon = np.array([w[2] for w in WAREHOUSES])
    h_lat = np.array([h[1] for h in HOSPITALS])
    h_lon = np.array([h[2] for h in HOSPITALS])
    dist = haversine_km(w_lat[:, None], w_lon[:, None], h_lat[None, :], h_lon[None, :])
    return Instance(
        warehouses=[w[0] for w in WAREHOUSES],
        hospitals=[h[0] for h in HOSPITALS],
        fixed_cost=np.array([w[3] for w in WAREHOUSES], dtype=float),
        capacity=np.array([w[4] for w in WAREHOUSES], dtype=float),
        base_demand=np.array([h[3] for h in HOSPITALS], dtype=float),
        transport_cost=dist * TRANSPORT_COST_PER_KM,
        distance_km=dist,
    )


def sample_scenarios(inst: Instance, n: int, seed: int,
                     demand_cv: float = 0.25, disruption_prob: float = 0.15,
                     disruption_availability: float = 0.2) -> list:
    """Demand uncertainty (lognormal, mean = base demand) plus random warehouse disruptions.

    A disruption means only `disruption_availability` of that warehouse's stock can be shipped
    (e.g. a power outage, a cold-chain failure or blocked roads).
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + demand_cv ** 2))
    mu = -sigma ** 2 / 2  # keeps the mean multiplier at 1
    scenarios = []
    for k in range(n):
        demand = np.round(inst.base_demand * rng.lognormal(mu, sigma, inst.H))
        availability = np.ones(inst.W)
        label = "demand only"
        if rng.random() < disruption_prob:
            w = rng.integers(inst.W)
            availability[w] = disruption_availability
            label = f"disruption at {inst.warehouses[w]}"
        scenarios.append(Scenario(1.0 / n, demand, availability, label))
    return scenarios


def expected_value_scenario(inst: Instance) -> list:
    """Single scenario with mean demand and no disruptions (what a deterministic planner assumes)."""
    return [Scenario(1.0, inst.base_demand.copy(), np.ones(inst.W), "expected value")]
