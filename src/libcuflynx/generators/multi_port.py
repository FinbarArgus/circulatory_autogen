"""Helpers for the per-variable (list-form) ``multi_port`` port setting.

A port in a module_config.json entry may carry ``multi_port``. The string forms
(``"True"``, ``"sum"``, ``"Sum"``) apply to the whole port and are handled where they
always were. The list form gives one entry per port variable, aligned with the port's
``variables``::

    {"port_type": "vessel_port", "variables": ["v_in", "u"], "multi_port": ["sum", "True"]}

* ``"sum"`` -- this module's variable (an input) is the sum, over every module connected
  through this port, of the neighbour's corresponding port variable.
* ``"True"`` -- this module's variable is mapped to the corresponding variable of every
  connected neighbour (normally an output: one source, many sinks).

A list-form port accepts any number of neighbours, on entrance, exit or general ports.
"""

MULTI_PORT_SUM = 'sum'
MULTI_PORT_SHARED = 'True'

# Entries accepted in a list-form multi_port, normalised to the two canonical values.
_ENTRY_ALIASES = {
    'sum': MULTI_PORT_SUM,
    'True': MULTI_PORT_SHARED,
    True: MULTI_PORT_SHARED,
}


def list_multi_port(port):
    """The normalised list-form multi_port of ``port``, or None if it has no list form."""
    if not isinstance(port, dict):
        return None
    value = port.get('multi_port')
    if not isinstance(value, (list, tuple)):
        return None
    return [_ENTRY_ALIASES.get(entry, entry) for entry in value]


def is_multi_port(port):
    """Whether ``port`` accepts connections from several modules ("True" or list form)."""
    if not isinstance(port, dict):
        return False
    value = port.get('multi_port')
    return value in ['True', True] or isinstance(value, (list, tuple))


def validate_list_multi_port(port, module_description):
    """Raise ValueError if a list-form multi_port on ``port`` is malformed.

    Ports without a list-form multi_port are not checked here.
    """
    value = port.get('multi_port') if isinstance(port, dict) else None
    if not isinstance(value, (list, tuple)):
        return
    variables = port.get('variables', [])
    if len(value) != len(variables):
        raise ValueError(
            f'{module_description}: port "{port.get("port_type")}" has a list-form multi_port '
            f'{list(value)} of length {len(value)}, but {len(variables)} variables '
            f'{list(variables)}. The list must have one entry per port variable.')
    for entry, variable in zip(value, variables):
        if entry not in _ENTRY_ALIASES:
            raise ValueError(
                f'{module_description}: port "{port.get("port_type")}" has multi_port entry '
                f'{entry!r} for variable "{variable}". List-form entries must be "sum" or "True".')


def module_has_list_multi_port(module_row):
    """Whether any entrance, exit or general port of a module row has a list-form multi_port."""
    for key in ('entrance_ports', 'exit_ports', 'general_ports'):
        ports = module_row[key] if key in module_row else []
        if isinstance(ports, (list, tuple)):
            for port in ports:
                if list_multi_port(port) is not None:
                    return True
    return False
