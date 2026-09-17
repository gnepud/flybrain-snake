"""Statistical hypothesis testing and metric calculation utilities.

Provides:
- welch_t_test: Two-sample Welch's t-test with Welch-Satterthwaite degrees of freedom.
- mann_whitney_u: Mann-Whitney U test with tie correction and normal approximation.
- Descriptive statistical helpers: mean, std_dev, median.
"""

import math
from typing import Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    """Calculate arithmetic mean of a sequence.
    
    Returns 0.0 for empty sequences.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: Sequence[float], ddof: int = 1) -> float:
    """Calculate sample standard deviation with specified degrees of freedom delta.
    
    Returns 0.0 if len(values) <= ddof.
    """
    n = len(values)
    if n <= ddof:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - ddof)
    return math.sqrt(max(0.0, var))


def median(values: Sequence[float]) -> float:
    """Calculate median value of a sequence.
    
    Returns 0.0 for empty sequences.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _incomplete_beta(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation if x > (a + 1)/(a + b + 2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _incomplete_beta(b, a, 1.0 - x, max_iter=max_iter, tol=tol)

    try:
        ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a
    except (ValueError, OverflowError):
        return 0.0

    # Lentz's continued fraction method
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = f * d

    for m in range(1, max_iter + 1):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f = f * c * d

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f = f * delta
        if abs(delta - 1.0) < tol:
            break

    return max(0.0, min(1.0, front * f))


def _student_t_p_value(t_stat: float, df: float) -> float:
    """Calculate two-tailed p-value for Student's t-statistic with df degrees of freedom."""
    if abs(t_stat) < 1e-15:
        return 1.0
    if df <= 0.0:
        return 1.0

    # For large degrees of freedom, normal approximation is fast and extremely accurate
    if df > 150.0:
        z = abs(t_stat)
        p_norm = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return max(0.0, min(1.0, p_norm))

    # Analytical Student-t distribution via regularized incomplete beta function
    # P(|T| >= |t|) = I_{df / (df + t^2)}(df/2, 1/2)
    x = df / (df + t_stat * t_stat)
    try:
        p_val = _incomplete_beta(df / 2.0, 0.5, x)
        return max(0.0, min(1.0, p_val))
    except Exception:
        # Fallback to normal approximation
        z = abs(t_stat)
        p_norm = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return max(0.0, min(1.0, p_norm))


def welch_t_test(s1: Sequence[float], s2: Sequence[float]) -> Tuple[float, float, float]:
    """Perform Welch's two-sample t-test for unequal variances.
    
    Args:
        s1: First sample values.
        s2: Second sample values.
        
    Returns:
        (t_statistic, p_value, degrees_of_freedom)
    """
    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0, 1.0

    m1 = sum(s1) / n1
    m2 = sum(s2) / n2

    var1 = sum((x - m1) ** 2 for x in s1) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in s2) / (n2 - 1)

    vn1 = var1 / n1
    vn2 = var2 / n2
    se = math.sqrt(vn1 + vn2)

    # Handle zero variance edge cases
    if se == 0.0:
        df_pooled = float(n1 + n2 - 2)
        if abs(m1 - m2) < 1e-15:
            return 0.0, 1.0, df_pooled
        t_inf = float("inf") if m1 > m2 else float("-inf")
        return t_inf, 0.0, df_pooled

    # Welch-Satterthwaite equation for degrees of freedom
    denom = ((vn1 ** 2) / (n1 - 1)) + ((vn2 ** 2) / (n2 - 1))
    if denom <= 0.0:
        df = float(n1 + n2 - 2)
    else:
        df = ((vn1 + vn2) ** 2) / denom

    t_stat = (m1 - m2) / se
    p_val = _student_t_p_value(t_stat, df)

    return t_stat, p_val, df


def mann_whitney_u(s1: Sequence[float], s2: Sequence[float]) -> Tuple[float, float]:
    """Perform two-sample Mann-Whitney U test with tie correction and normal approximation.
    
    Args:
        s1: First sample values.
        s2: Second sample values.
        
    Returns:
        (u_statistic, p_value)
    """
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine observations with sample origin tags
    combined = [(float(val), 1) for val in s1] + [(float(val), 2) for val in s2]
    combined.sort(key=lambda item: item[0])

    # Assign average ranks to tied groups
    n_tot = n1 + n2
    ranks = [0.0] * n_tot
    tie_counts = []
    i = 0
    while i < n_tot:
        j = i
        while j < n_tot and combined[j][0] == combined[i][0]:
            j += 1
        tie_len = j - i
        if tie_len > 1:
            tie_counts.append(tie_len)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for sample 1
    r1 = sum(ranks[k] for k in range(n_tot) if combined[k][1] == 1)
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    if n_tot < 2:
        return u, 1.0

    # Variance calculation with tie correction:
    # Var(U) = (n1 * n2 / 12) * [ (n1 + n2 + 1) - sum(t^3 - t) / ((n1 + n2)(n1 + n2 - 1)) ]
    tie_sum = sum(t ** 3 - t for t in tie_counts)
    tie_adj = tie_sum / (n_tot * (n_tot - 1)) if n_tot > 1 else 0.0
    var_u = (n1 * n2 / 12.0) * ((n_tot + 1) - tie_adj)

    if var_u <= 0.0:
        return u, 1.0

    sigma = math.sqrt(var_u)
    mu = (n1 * n2) / 2.0
    z = (u - mu) / sigma
    p_val = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    return u, max(0.0, min(1.0, p_val))
