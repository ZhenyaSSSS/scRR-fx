import torch
from mpmath import mp
import pytest

from scrr_fx import two_sum
from scrr_fx._core import two_prod

mp.dps = 200


@pytest.mark.parametrize("exp", [10, 50, 100, 200])
def test_catastrophic_cancellation(exp):
    # a ~= -b, разница ~1e-<exp>
    a_val = 1.0
    b_val = -1.0 + 10 ** (-exp)
    a = torch.tensor(a_val, dtype=torch.float64)
    b = torch.tensor(b_val, dtype=torch.float64)
    s, e = two_sum(a, b)

    gt = mp.mpf(a_val) + mp.mpf(b_val)
    scrr = mp.mpf(s.item()) + mp.mpf(e.item())
    assert mp.almosteq(gt, scrr)


@pytest.mark.parametrize(
    "a,b",
    [
        (1e38, 1e38),
        (1.0, 1e-38),
        (1e18, 1),
        (1.0, -1.0),
        (torch.pi, torch.e),
    ],
)
def test_two_sum_edge_cases(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    s, e = two_sum(a, b)
    assert (a + b) == (s + e)


@pytest.mark.parametrize(
    "a,b",
    [
        (1e200, 1e200),
        (1e-200, 1e-200),
        (torch.finfo(torch.float64).max, 2.0),
    ],
)
def test_two_prod_edge_cases(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    p, e = two_prod(a, b)
    if torch.isinf(a * b):
        assert torch.isinf(p)
    else:
        assert pytest.approx((a * b).item()) == (p + e).item()


@pytest.mark.parametrize(
    "a, b, s_is_nan, e_is_nan",
    [
        (torch.nan, 1.0, True, True),
        (1.0, torch.nan, True, True),
        (torch.nan, torch.nan, True, True),
        (torch.inf, torch.nan, True, True),
        (torch.inf, 1.0, False, False),
        (torch.inf, torch.inf, False, False),
        (torch.inf, -torch.inf, True, False),
    ]
)
def test_two_sum_non_finite(a, b, s_is_nan, e_is_nan):
    """Проверяет поведение two_sum с не-конечными числами (NaN, Inf)."""
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    
    s, e = two_sum(a, b)
    
    assert torch.isnan(s).item() == s_is_nan
    # Компонент ошибки 'e' не всегда NaN, когда 's' является NaN.
    # Поэтому мы проверяем e_is_nan только тогда, когда это ожидается.
    if e_is_nan:
        assert torch.isnan(e).item()


@pytest.mark.parametrize(
    "a, b, p_is_nan, e_is_nan",
    [
        (torch.nan, 1.0, True, True),
        (1.0, torch.nan, True, True),
        (torch.nan, torch.nan, True, True),
        (torch.inf, torch.nan, True, True),
        (torch.inf, 1.0, False, False),
        (torch.inf, torch.inf, False, False),
        (torch.inf, 0.0, True, True),
    ]
)
def test_two_prod_non_finite(a, b, p_is_nan, e_is_nan):
    """Проверяет поведение two_prod с не-конечными числами (NaN, Inf)."""
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    
    p, e = two_prod(a, b)
    
    assert torch.isnan(p).item() == p_is_nan
    if e_is_nan:
        assert torch.isnan(e).item()

    # Проверка на inf, где это применимо
    if not p_is_nan and (torch.isinf(a) or torch.isinf(b)):
         assert torch.isinf(p) 