import contextlib
import platform
import logging
import sys

import numpy as np
import os

from logging.handlers import RotatingFileHandler
from logging import StreamHandler
date_format = "%Y-%m-%d-%H-%M-%S"

@contextlib.contextmanager
def local_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_device_placement():
    return os.getenv("ATES_DEVICE_PLACEMENT", "CPU")

def get_logger_instance(filename):
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)

    has_stdout = False
    has_file = False
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            has_file = True
        if isinstance(handler, StreamHandler):
            has_stdout = True

    if not has_stdout:
        sh = StreamHandler(sys.stdout)
        sh.addFilter(HostnameFilter())
        root_logger.addHandler(sh)

    if filename is not None and not has_file:
        fh = RotatingFileHandler(filename,
                                 maxBytes=32*1024*1024,
                                 backupCount=10)
        fh.addFilter(HostnameFilter())
        formatter = logging.Formatter(fmt='%(hostname)s %(asctime)s - PID%(process)d %(message)s', datefmt=date_format)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger


class HostnameFilter(logging.Filter):
    hostname = platform.node()
    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True