"""lilytorch — GPU-accelerated CFD with immersed-boundary methods."""

import logging

__version__ = "0.1.0"


def configure_logging(level: int = logging.WARNING) -> None:
    """Set up logging for the entire lilytorch package.

    Call this once in your script or notebook before running a simulation::

        import lilytorch
        lilytorch.configure_logging(logging.INFO)

    Parameters
    ----------
    level : int
        Root log level for the ``lilytorch`` logger hierarchy.
        Common choices: ``logging.DEBUG``, ``logging.INFO``,
        ``logging.WARNING`` (default), ``logging.ERROR``.
    """
    pkg_logger = logging.getLogger("lilytorch")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
