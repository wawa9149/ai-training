import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d: %(name)-20s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
