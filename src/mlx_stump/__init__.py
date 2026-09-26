"""mlx-stump: matrix profile on Apple Silicon GPUs, STUMPY-compatible API.

STUMPY is a trademark of TD Ameritrade IP Company, Inc. mlx-stump is an
independent project and is not affiliated with or endorsed by the STUMPY
project or TD Ameritrade.
"""

from ._mass import mass, mass_absolute
from ._match import aamp_match, match
from ._mparray import mparray
from ._stump import stump

__version__ = "0.1.0.dev0"

__all__ = [
    "stump",
    "mass",
    "mass_absolute",
    "match",
    "aamp_match",
    "mparray",
    "__version__",
]
