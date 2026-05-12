"""Marking and checking functions that are safe for CasADi symbolic execution."""

CIRCULATORY_DIFFERENTIABLE = "_circulatory_differentiable"


def differentiable(func):
    setattr(func, CIRCULATORY_DIFFERENTIABLE, True)
    return func


def is_circulatory_differentiable(func):
    return getattr(func, CIRCULATORY_DIFFERENTIABLE, False)


def assert_casadi_differentiable(obs_info, cost_type, operation_funcs_dict, cost_funcs_dict):
    """
    Raise ValueError if any operation or cost referenced by obs_info / cost_type
    is not marked with @differentiable.
    """
    if obs_info is not None:
        ops = obs_info.get("operations") or []
        for name in dict.fromkeys(ops):
            if name is None:
                continue
            fn = operation_funcs_dict.get(name)
            if fn is None:
                raise ValueError(f"Unknown operation {name!r} in obs_info['operations'].")
            if not is_circulatory_differentiable(fn):
                raise ValueError(
                    f"Operation {name!r} is used in casadi_python mode but is not marked "
                    f"@differentiable (not safe for symbolic execution)."
                )
    if cost_type is not None and cost_funcs_dict is not None:
        raw = cost_type if isinstance(cost_type, (list, tuple)) else [cost_type]
        names = list(dict.fromkeys(raw))
        for name in names:
            if name is None:
                continue
            fn = cost_funcs_dict.get(name)
            if fn is None:
                raise ValueError(f"Unknown cost function {name!r} in cost_type.")
            if not is_circulatory_differentiable(fn):
                raise ValueError(
                    f"Cost function {name!r} is used in casadi_python mode but is not marked "
                    f"@differentiable."
                )
