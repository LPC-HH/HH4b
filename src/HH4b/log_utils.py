## https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
from __future__ import annotations

log_config = {
    "version": 1,
    "disable_existing_loggers": False,  # this fixes the problem about many loggers
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)-2s: %(name)-20s [%(threadName)s] %(message)s"
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}
