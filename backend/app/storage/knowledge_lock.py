"""Serialize local knowledge publication and recycle-bin moves."""
from functools import wraps
from threading import RLock

_lock = RLock()


def knowledge_mutation(function):
    @wraps(function)
    def protected(*args, **kwargs):
        with _lock:
            return function(*args, **kwargs)
    return protected
