from .simple_lama import SimpleLama, main, main_cli

__all__ = [
    "SimpleLama",
    "main",
    "main_cli"
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
