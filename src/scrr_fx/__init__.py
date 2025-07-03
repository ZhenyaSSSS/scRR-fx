"""SCRR-FX — Stream-Compensated Residual Representation Framework.

На текущем этапе экспортируются только низкоуровневые EFT-примитивы
для тестирования. Более поздние стадии добавят Renormalize, SCRR_Tensor
и высокоуровневые операции.
"""

from ._core import two_prod, two_sum  # noqa: F401
from ._renorm import renormalize  # noqa: F401
from ._tensor import SCRR_Tensor  # noqa: F401

# Импортируем _ops ради побочных эффектов (рег. HANDLED_FUNCTIONS)
from . import _ops as _scrr_ops  # noqa: F401

__all__ = [
    "two_sum",
    "two_prod",
    "renormalize",
    "SCRR_Tensor",
    # базовые операции (публичный API)
    "_scrr_ops",
] 