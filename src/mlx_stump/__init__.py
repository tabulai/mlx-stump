"""mlx-stump: matrix profile on Apple Silicon GPUs, STUMPY-compatible API.

STUMPY is a trademark of TD Ameritrade IP Company, Inc. mlx-stump is an
independent project and is not affiliated with or endorsed by the STUMPY
project or TD Ameritrade.
"""

from ._engine import estimated_peak_bytes
from ._mass import mass
from ._match import match
from ._mparray import mparray
from ._stump import aamp, gpu_aamp, gpu_stump, stump

__version__ = "0.1.0.dev0"

__all__ = [
    "stump",
    "aamp",
    "gpu_stump",
    "gpu_aamp",
    "mass",
    "match",
    "mparray",
    "estimated_peak_bytes",
    "__version__",
]
