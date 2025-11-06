import subprocess
import shlex
import tempfile
import uuid
import inspect
from copy import deepcopy
from functools import wraps

from invoke import task as _task
from invoke.context import Context

sbatch_args = [
    "account", "job_name", "comment", "workdir", "chdir",
    "begin", "deadline", "time", "time_min", "dependency", "priority",
    "cpus_per_task", "cpus_per_gpu", "mem", "mem_per_cpu", "mem_per_gpu", "mem_bind",
    "ntasks", "ntasks_per_core", "ntasks_per_node", "ntasks_per_socket",
    "nodes", "node", "nodelist", "exclude", "constraint", "cluster_constraint",
    "sockets_per_node", "cores_per_socket", "threads_per_core",
    "gres", "gres_flags", "gpu_bind", "gpu_freq", "gpus", "gpus_per_node", 
    "gpus_per_socket", "gpus_per_task",
    "partition", "qos", "reservation", "wckey",
    "input", "output", "error", "open_mode",
    "array",
    "mail_type", "mail_user",
    "uid", "gid", "user",
    "export", "export_file", "environment",
    "network", "switch", "switches",
    "profile", "acctg_freq", "bb", "bbf", "burst_buffer", "clusters",
    "core_spec", "cpu_freq", "distribution",
    "geometry", "hint", "mcs_label", "nodefile",
    "pack_group", "power",
    "signal", "thread_spec",
    "tmp", "usage", "wait",
    "licenses",
    "container", "container_id",
    "het_group",
    "x11"
]
reserved_args = sbatch_args + ["run_locally"]

# Kwargs are for setting default values for sbatch arguments or task kwargs
def slurm_task(*task_args, env_path: str | None = None, **kwargs):
    def _slurm_task(func):
        # Change the signature of the function to include the `sbatch` arguments
        func_sig = inspect.signature(func)
        func_params = list(func_sig.parameters.values())
        func_param_names = [p.name for p in func_params]

        for name in func_param_names:
            if name in reserved_args:
                raise ValueError(f"parameter name `{name}` is a reserved argument name")

        wrapped_func_params = deepcopy(func_params)
        wrapped_func_params.append(inspect.Parameter("run_locally", inspect.Parameter.KEYWORD_ONLY, default=False))

        for name in sbatch_args:
            wrapped_func_params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None))

        wrapped_func_sig = func_sig.replace(parameters=wrapped_func_params)

        @wraps(func)
        def wrapped_func(*func_args, **wrapped_func_kwargs):
            bound_args = wrapped_func_sig.bind(*func_args, **wrapped_func_kwargs)
            bound_args.apply_defaults()

            run_locally = bound_args.arguments["run_locally"]
            func_arg_dict = {k: v for k, v in wrapped_func_kwargs.items() if k in func_param_names}

            if run_locally:
                return func(*func_args, **func_arg_dict)

            # Update `sbatch_config_dict` with `sbatch` arguments
            for key, val in bound_args.arguments.items():
                if val and key in sbatch_args:
                    sbatch_config_dict[key] = val

            # Filter out `Context` object, `sbatch` arguments, and `None` values from things to pass to `invoke`
            func_arg_dict = {}

            for key, val in bound_args.arguments.items():
                if key in func_param_names and not isinstance(val, Context) and val is not None:
                    func_arg_dict[key] = val

            job_script_str = generate_job_script_str(func, sbatch_config_dict, func_arg_dict, env_path)

            # Write `job_script_str` to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
            job_script_path = temp_file.name

            with open(job_script_path, "w") as f:
                f.write(job_script_str)

            subprocess.run(shlex.split(f"sbatch {job_script_path}"))

        wrapped_func.__signature__ = wrapped_func_sig

        # Create a wrapper function that ignores the context parameter and handles invoke arguments
        def context_wrapper(context, **kwargs):
            # Filter kwargs to only include those that wrapped_func expects
            filtered_kwargs = {}
            for param_name in [p.name for p in wrapped_func_params]:
                if param_name in kwargs:
                    filtered_kwargs[param_name] = kwargs[param_name]
            
            # Add any SLURM arguments that were passed
            for arg_name in sbatch_args + ["run_locally"]:
                if arg_name in kwargs:
                    filtered_kwargs[arg_name] = kwargs[arg_name]
            
            return wrapped_func(**filtered_kwargs)
        
        context_wrapper.__name__ = wrapped_func.__name__
        context_wrapper.__doc__ = wrapped_func.__doc__
        context_wrapper.__module__ = wrapped_func.__module__
        
        # Set the signature of context_wrapper to match what invoke expects
        # This should include context as first parameter, then all the SLURM parameters
        context_param = inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        all_params = [context_param] + list(wrapped_func_sig.parameters.values())
        context_wrapper.__signature__ = inspect.Signature(all_params)

        return task_decorator(context_wrapper)

    task_kwargs = {}
    sbatch_config_dict = {}

    for key, val in kwargs.items():
        if key in sbatch_args:
            sbatch_config_dict[key] = val
        else:
            task_kwargs[key] = val

    # This is for when the decorator is used without parentheses
    # In order to mimic the behavior of `@task`
    if len(task_args) == 1 and callable(task_args[0]) and len(task_kwargs) == 0:
        task_decorator = _task
        return _slurm_task(task_args[0])
    else:
        task_decorator = _task(*task_args, **task_kwargs)
        return _slurm_task


# The `func_arg_dict` has the args passed to `invoke`
def generate_job_script_str(func, sbatch_config_dict: dict[str, str], func_arg_dict: dict[str, str | bool] | None = None, env_path: str | None = None):
    if not func_arg_dict:
        func_arg_dict = {}
    
    job_script_str = "#!/bin/bash\n"

    job_name = sbatch_config_dict.get("job_name", f"job_{uuid.uuid4()}")
    sbatch_config_dict["job_name"] = job_name

    for key, val in sbatch_config_dict.items():
        arg_name = key.replace("_", "-")
        job_script_str += f"#SBATCH --{arg_name}={val}\n"

    job_script_str += "module load python\n"

    if env_path:
        job_script_str += f"source {env_path}\n"

    task_name = func.__name__.replace("_", "-")
    format_arg = lambda x: f"\"{x}\"" if isinstance(x, str) else x # Need to wrap strings in quotes
    arg_str = ""

    for key, val in func_arg_dict.items():
        if isinstance(val, str):
            arg_str += f"--{key.replace('_', '-')} {format_arg(val)} "
        elif isinstance(val, bool):
            arg_str += f"--{key.replace('_', '-')} " if val else ""
        else:
            arg_str += f"--{key.replace('_', '-')} {val} "
    
    job_script_str += f"invoke {task_name} --run-locally {arg_str}\n"

    return job_script_str