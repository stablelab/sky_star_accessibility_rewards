import logging

logger = logging.getLogger("stablelab-logger")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s() - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler) 