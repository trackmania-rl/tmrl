#!/usr/bin/env python3

import logging


LOG_FORMAT = "[%(levelname)-5s] - %(name)-15s - %(funcName)15s() - %(message)s"
LOG_LEVEL = logging.INFO


def setup_logger(logger):
    for handler in logger.handlers:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(LOG_LEVEL)
