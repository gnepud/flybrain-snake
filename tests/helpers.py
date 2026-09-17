"""Shared test helpers, reference mathematical oracles, and dynamic import utilities."""

import importlib
import math
import sys
import unittest
from typing import Any, Callable


def safe_import(module_path: str, *symbols: str) -> tuple[Any, ...]:
    """Safely import a module or specific symbols from src.
    
    Returns a tuple of imported objects or None if not yet implemented.
    """
    try:
        mod = importlib.import_module(module_path)
        if not symbols:
            return (mod,)
        res = []
        for sym in symbols:
            res.append(getattr(mod, sym, None))
        return tuple(res)
    except (ImportError, ModuleNotFoundError):
        return tuple(None for _ in (symbols or [None]))


def require_symbols(*imported_objects, feature_desc: str = "Feature"):
    """Decorator or check that skips a test if required symbols are not yet available."""
    missing = [obj for obj in imported_objects if obj is None]
    if missing:
        raise unittest.SkipTest(f"{feature_desc} is not yet implemented (waiting for milestone)")


# Reference mathematical oracles derived from specifications
def reference_relative_bearing(head: tuple[int, int], food: tuple[int, int], direction: int) -> float:
    """Calculate expected relative bearing angle in radians in [-pi, pi].
    
    direction: 0=UP(0, -1), 1=RIGHT(1, 0), 2=DOWN(0, 1), 3=LEFT(-1, 0)
    """
    dx = food[0] - head[0]
    dy = food[1] - head[1]
    if dx == 0 and dy == 0:
        return 0.0

    target_angle = math.atan2(dy, dx)
    heading_angles = {
        0: -math.pi / 2,   # UP (0, -1)
        1: 0.0,            # RIGHT (1, 0)
        2: math.pi / 2,    # DOWN (0, 1)
        3: math.pi,        # LEFT (-1, 0)
    }
    head_angle = heading_angles[direction]
    diff = target_angle - head_angle
    # Normalize to [-pi, pi]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    return diff


def reference_euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean distance between two grid points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def reference_welch_t_test(s1: list[float], s2: list[float]) -> tuple[float, float, float]:
    """Pure Python reference implementation of Welch's two-sample t-test."""
    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0, 1.0

    mean1 = sum(s1) / n1
    mean2 = sum(s2) / n2
    var1 = sum((x - mean1) ** 2 for x in s1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in s2) / (n2 - 1)

    vn1 = var1 / n1
    vn2 = var2 / n2
    se = math.sqrt(vn1 + vn2)
    if se == 0:
        return 0.0, 1.0, 1.0

    t_stat = (mean1 - mean2) / se
    df = (vn1 + vn2) ** 2 / ((vn1 ** 2) / (n1 - 1) + (vn2 ** 2) / (n2 - 1))

    # Approximate 2-tailed p-value via normal CDF for moderate/large df
    z = abs(t_stat)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return t_stat, max(0.0, min(1.0, p_val)), df


def reference_mann_whitney_u(s1: list[float], s2: list[float]) -> tuple[float, float]:
    """Pure Python reference implementation of Mann-Whitney U test."""
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    combined = [(val, 1) for val in s1] + [(val, 2) for val in s2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks with ties averaged
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for p-value
    mu = (n1 * n2) / 2.0
    sigma = math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    p_val = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return u, max(0.0, min(1.0, p_val))
