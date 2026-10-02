'''
Reads module configs and vessel arrays written in either of the two schemas in use, and
converts them to libcuflynx's internal names.

Module config entries (``*_modules_config.json``):

==================  ==================  ============================================
libcuflynx          PhLynx              meaning
==================  ==================  ============================================
``vessel_type``     ``module_type``     the kind of module named in a vessel array
``BC_type``         ``module_subtype``  its boundary-condition variant
``module_file``     ``component_file``  the CellML file the module lives in
``module_type``     ``component_type``  the CellML component name
==================  ==================  ============================================

``module_type`` means different things in the two schemas, so the PhLynx schema is recognised
by ``module_subtype``, ``component_file`` or ``component_type``, never by ``module_type``. An
entry that mixes keys of both schemas is an error. One library may use both schemas across
(or even within) its files: each entry is read on its own.

Vessel arrays (``<prefix>_vessel_array.csv``, or PhLynx's ``<prefix>_module_array.csv``):

* libcuflynx: ``name, BC_type, vessel_type, inp_vessels, out_vessels``
* PhLynx's Circulatory Autogen export: ``name, module_type, module_subtype, inp_instances,
  out_instances`` (``module_type`` is the vessel_type, ``module_subtype`` the BC_type).

Normalisation happens where the files are loaded, so everything downstream only ever sees
the libcuflynx names.
'''

import json
import os

import pandas as pd

from libcuflynx.generators.multi_port import normalise_port_multi_port

# PhLynx key -> libcuflynx key, for module config entries
PHLYNX_MODULE_KEYS = {
    'module_type': 'vessel_type',
    'module_subtype': 'BC_type',
    'component_file': 'module_file',
    'component_type': 'module_type',
}
# keys that only a PhLynx-schema entry has
_PHLYNX_ONLY_KEYS = ('module_subtype', 'component_file', 'component_type')
# keys that only a libcuflynx-schema entry has
_LIBCUFLYNX_ONLY_KEYS = ('vessel_type', 'BC_type', 'module_file')

_PORT_KEYS = ('entrance_ports', 'exit_ports', 'general_ports')

# PhLynx column -> libcuflynx column, for vessel arrays
PHLYNX_VESSEL_ARRAY_COLUMNS = {
    'module_type': 'vessel_type',
    'module_subtype': 'BC_type',
    'inp_instances': 'inp_vessels',
    'out_instances': 'out_vessels',
}
_PHLYNX_ONLY_COLUMNS = ('module_subtype', 'inp_instances', 'out_instances')
_LIBCUFLYNX_ONLY_COLUMNS = ('BC_type', 'vessel_type', 'inp_vessels', 'out_vessels')
_LIBCUFLYNX_COLUMN_ORDER = ('name', 'BC_type', 'vessel_type', 'inp_vessels', 'out_vessels')


def _describe(entry, source):
    where = f' in {source}' if source else ''
    name = entry.get('vessel_type', entry.get('module_type', '?'))
    subtype = entry.get('BC_type', entry.get('module_subtype', '?'))
    return f'module config entry ({name}, {subtype}){where}'


def normalise_module_config_entry(entry, source=None):
    '''
    One module config entry, in either schema, as a libcuflynx-schema entry.

    A libcuflynx entry comes back with its keys unchanged; a PhLynx entry has its keys
    renamed, with ``vessel_type`` and ``BC_type`` first and the rest in their original order.
    In both, ``multi_port`` values are put in canonical form (see
    ``generators/multi_port.normalise_port_multi_port``). ``entry`` itself is not modified.

    Raises ValueError for an entry that mixes the two schemas, a PhLynx entry missing one of
    its four keys, or an entry in neither schema.
    '''
    if not isinstance(entry, dict):
        raise ValueError(f'module config entry{" in " + source if source else ""} is a '
                         f'{type(entry).__name__}, not a JSON object: {entry!r}')
    phlynx_keys = [k for k in _PHLYNX_ONLY_KEYS if k in entry]
    libcuflynx_keys = [k for k in _LIBCUFLYNX_ONLY_KEYS if k in entry]
    if phlynx_keys and libcuflynx_keys:
        raise ValueError(
            f'{_describe(entry, source)} mixes the libcuflynx schema (has {libcuflynx_keys}) '
            f'and the PhLynx schema (has {phlynx_keys}). Use one schema per entry: either '
            f'vessel_type/BC_type/module_file/module_type, or '
            f'module_type/module_subtype/component_file/component_type.')

    if phlynx_keys:
        missing = [k for k in PHLYNX_MODULE_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f'{_describe(entry, source)} is in the PhLynx schema (has {phlynx_keys}) but '
                f'is missing {missing}. A PhLynx-schema entry needs module_type, '
                f'module_subtype, component_file and component_type.')
        renamed = {PHLYNX_MODULE_KEYS.get(k, k): v for k, v in entry.items()}
        normalised = {'vessel_type': renamed.pop('vessel_type'), 'BC_type': renamed.pop('BC_type')}
        normalised.update(renamed)
    elif libcuflynx_keys:
        normalised = dict(entry)
    else:
        raise ValueError(
            f'Cannot tell the schema of {_describe(entry, source)}: it has none of '
            f'vessel_type/BC_type/module_file (libcuflynx) or '
            f'module_subtype/component_file/component_type (PhLynx). Keys: {list(entry)}')

    description = _describe(normalised, source)
    for port_key in _PORT_KEYS:
        ports = normalised.get(port_key)
        if isinstance(ports, list):
            normalised[port_key] = [normalise_port_multi_port(port, description) for port in ports]
    return normalised


def normalise_module_config(entries, source=None):
    '''A list of module config entries (one JSON file), each normalised on its own.'''
    if not isinstance(entries, list):
        raise ValueError(f'module config {source or ""} must be a JSON list of entries, '
                         f'not a {type(entries).__name__}')
    return [normalise_module_config_entry(entry, source) for entry in entries]


def load_module_config(path):
    '''The normalised entries of the module config JSON file at ``path``.'''
    with open(path, encoding='utf-8-sig') as f:
        return normalise_module_config(json.load(f), source=str(path))


def normalise_vessel_array_columns(df, source=None):
    '''
    A vessel-array dataframe with libcuflynx column names.

    A PhLynx-layout array (``module_type``, ``module_subtype``, ``inp_instances``,
    ``out_instances``) has its columns renamed and put in the libcuflynx order (``name,
    BC_type, vessel_type, inp_vessels, out_vessels``, then any others). A libcuflynx-layout
    array is returned unchanged. Column names are expected to be stripped already.
    '''
    columns = [str(c) for c in df.columns]
    phlynx_columns = [c for c in _PHLYNX_ONLY_COLUMNS if c in columns]
    libcuflynx_columns = [c for c in _LIBCUFLYNX_ONLY_COLUMNS if c in columns]
    where = f' {source}' if source else ''
    if phlynx_columns and libcuflynx_columns:
        raise ValueError(
            f'vessel array{where} mixes libcuflynx columns {libcuflynx_columns} and PhLynx '
            f'columns {phlynx_columns}. Use either name,BC_type,vessel_type,inp_vessels,'
            f'out_vessels or name,module_type,module_subtype,inp_instances,out_instances.')
    if not phlynx_columns:
        return df
    missing = [c for c in ('name',) + tuple(PHLYNX_VESSEL_ARRAY_COLUMNS) if c not in columns]
    if missing:
        raise ValueError(f'vessel array{where} is in the PhLynx layout (has {phlynx_columns}) '
                         f'but is missing the columns {missing}.')
    df = df.rename(columns=PHLYNX_VESSEL_ARRAY_COLUMNS)
    ordered = list(_LIBCUFLYNX_COLUMN_ORDER) + [c for c in df.columns if c not in _LIBCUFLYNX_COLUMN_ORDER]
    return df[ordered]


def read_vessel_array_csv(path, **read_csv_kwargs):
    '''``pd.read_csv`` of a vessel array, with its columns normalised to libcuflynx names.'''
    df = pd.read_csv(path, **read_csv_kwargs)
    df = df.rename(columns=lambda c: str(c).strip())
    return normalise_vessel_array_columns(df, source=str(path))


def vessel_array_path(resources_dir, file_prefix):
    '''
    The vessel array of ``file_prefix`` in ``resources_dir``: ``<prefix>_vessel_array.csv``,
    or, only if that does not exist, PhLynx's ``<prefix>_module_array.csv``. When neither
    exists the ``_vessel_array.csv`` path is returned, so the error names the usual file.
    '''
    vessel_array = os.path.join(resources_dir, file_prefix + '_vessel_array.csv')
    module_array = os.path.join(resources_dir, file_prefix + '_module_array.csv')
    if not os.path.exists(vessel_array) and os.path.exists(module_array):
        return module_array
    return vessel_array
