from .newton_raphson import solve as newton_raphson
from .bisection import solve as bisection
from .brent import solve as brent

__all__ = ["newton_raphson", "bisection", "brent"]
