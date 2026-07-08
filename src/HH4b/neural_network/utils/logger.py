from __future__ import annotations

import logging
import sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


COLORS = {
    "WARNING": YELLOW,
    "INFO": WHITE,
    "DEBUG": BLUE,
    "CRITICAL": YELLOW,
    "ERROR": RED,
}


def setup_logger(
    name: str | None = None,
    log_file: str | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Base message format
    message_format = (
        "[%(name)-10s][%(asctime)s][%(levelname)-10s]  %(message)s (%(filename)s:%(lineno)d)"
    )

    # File handler (no color or styling)
    if log_file:
        file_formatter = logging.Formatter(message_format)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler (with color)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    message = "[$BOLD%(name)-10s$RESET][%(asctime)s][%(levelname)-10s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    color_format = formatter_message(message, use_color=True)
    console_handler.setFormatter(ColoredFormatter(color_format))
    logger.addHandler(console_handler)

    return logger


def configure_logger(
    logger: logging.Logger,
    name: str | None = None,
    log_file: str | None = None,
    log_level: str | None = None,
) -> None:
    """
    Configure or reconfigure a logger with the specified settings.
    Only updates the parameters that are provided, keeping existing configuration otherwise.
    """
    # Store existing handlers configuration
    existing_handlers = {"file": None, "console": None, "current_level": logger.level}

    # Find and store existing handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing_handlers["file"] = handler.baseFilename
        elif isinstance(handler, logging.StreamHandler):
            existing_handlers["console"] = handler

    # Clear existing handlers
    logger.handlers.clear()

    # Update name if provided
    if name is not None:
        logger.name = name

    # Update log level only if provided
    if log_level is not None:
        logger.setLevel(log_level)

    # Base message format
    message_format = (
        "[%(name)-10s][%(asctime)s][%(levelname)-10s]  %(message)s (%(filename)s:%(lineno)d)"
    )

    # File handler (no color or styling)
    # Use new log_file if provided, otherwise use existing one if there was one
    file_to_use = log_file if log_file is not None else existing_handlers["file"]
    if file_to_use:
        file_formatter = logging.Formatter(message_format)
        file_handler = logging.FileHandler(file_to_use)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler (with color)
    console_handler = logging.StreamHandler(sys.stdout)
    # Use new log level if provided, otherwise use the logger's current level
    console_handler.setLevel(log_level if log_level is not None else logger.level)
    message = "[$BOLD%(name)-10s$RESET][%(asctime)s][%(levelname)-10s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    color_format = formatter_message(message, use_color=True)
    console_handler.setFormatter(ColoredFormatter(color_format))
    logger.addHandler(console_handler)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg: str, use_color: bool = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def formatter_message(message: str, use_color: bool = True) -> str:
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


LOGGER = setup_logger(name="HH4b Training", log_level="INFO")
