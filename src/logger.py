import logging
import sys
import time
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[95m",
        "INFO": "\033[94m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[1;91m",
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")

        timestamp = self.formatTime(
            record, "%H:%M:%S"
        )

        level = record.levelname
        source = record.name

        message = record.getMessage()

        return (
            f"{color}[{level}:{timestamp}] "
            f"[{source}] {message}{self.RESET}"
        )


def setup_logging(debug: bool = False):
    """Enable log output only when the CLI debug flag is set."""
    logging.disable(logging.NOTSET if debug else logging.CRITICAL)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def log_chunk_info(logger: logging.Logger, path: Path, start: int, end: int):
    """Log information about a specific chunk of text."""
    logger.debug(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] \n\tChunk path {path} [{start}:{end}] \n\tRun to get content: grep -o '.\{{{start},{end}}}' {path} \n\tRun to get content between two specific characters/strings: sed -n '/{start}/,/{end}/p' {path}")
