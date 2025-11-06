import os
import logging
from functools import wraps

import yaml
from invoke import task as _task

from utils.slurm import slurm_task as _slurm_task

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def task(func):
    """ Task decorator that handles context parameter automatically """
    import inspect
    from invoke import Context
    
    # Get the original signature
    sig = inspect.signature(func)
    
    params = [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Context)]
    params.extend(sig.parameters.values())
    new_sig = sig.replace(parameters=params)
    
    def wrapped_func(ctx, *args, **kwargs):
        return func(*args, **kwargs)
    
    wrapped_func.__name__ = func.__name__
    wrapped_func.__doc__ = func.__doc__
    wrapped_func.__module__ = func.__module__
    wrapped_func.__signature__ = new_sig
    
    return _task(wrapped_func)

def slurm_task(*args, **kwargs):
    kwargs.update({k: config["slurm"][k] for k in config["slurm"] if k not in kwargs})
    return _slurm_task(*args, **kwargs)