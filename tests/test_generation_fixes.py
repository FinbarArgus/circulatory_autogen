"""
Regression tests for generation bugs:

- #526: ``model_type: python`` code calls libCellML helpers (``eq_func``, ``or_func``,
  ``min``, ``sec``, ...) that the generated utilities never defined.
- #525: a ``sum`` multi-port (e.g. ``volume_sum``) with no inputs raised ``IndexError``.
"""
import math
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

REPO_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')


def _read_resource(filename):
    with open(os.path.join(REPO_RESOURCES_DIR, filename)) as f:
        return f.read()


def _generate(tmp_path, prefix, vessel_array, parameters, model_type='cellml',
              external_modules_dir=None):
    """Write ``{prefix}_vessel_array.csv`` / ``_parameters.csv`` and generate the model.

    Returns the directory the model was generated into.
    """
    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir(exist_ok=True)
    (resources_dir / f'{prefix}_vessel_array.csv').write_text(textwrap.dedent(vessel_array))
    (resources_dir / f'{prefix}_parameters.csv').write_text(textwrap.dedent(parameters))
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': model_type,
        'solver': 'solve_ivp' if model_type == 'python' else 'CVODE_myokit',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'external_modules_dir': external_modules_dir,
        'DEBUG': False,
    }
    if model_type == 'python':
        config['solver_info'] = {'method': 'RK45', 'max_step': 0.01}
    assert generate_with_new_architecture(False, config)
    return tmp_path / 'generated_models' / prefix


def _run(model_path, variables, model_type='cellml', sim_time=1.0, dt=0.01):
    """Run a generated model; return (t, {variable: series})."""
    if model_type == 'python':
        helper = get_simulation_helper(model_path=str(model_path), solver='solve_ivp',
                                       model_type='python', dt=dt, sim_time=sim_time,
                                       solver_info={'method': 'RK45', 'max_step': dt,
                                                    'rtol': 1e-8, 'atol': 1e-10},
                                       pre_time=0.0)
    else:
        helper = get_simulation_helper(model_path=str(model_path), solver='CVODE_myokit',
                                       model_type='cellml', dt=dt, sim_time=sim_time,
                                       pre_time=0.0)
    helper.run()
    results = helper.get_results([[v] for v in variables], flatten=True)
    t = np.asarray(helper.get_time())
    return t, {v: np.asarray(r).ravel() for v, r in zip(variables, results)}


# ---------------------------------------------------------------------------------------
# #526: every libCellML Python-profile helper is defined for model_type python
# ---------------------------------------------------------------------------------------

LOGIC_MODULES_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="eq_rate_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="k" public_interface="in" units="dimensionless"/>
        <variable initial_value="0" name="x" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
                <piecewise>
                    <piece>
                        <cn cellml:units="per_s">1</cn>
                        <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">1</cn></apply>
                    </piece>
                    <otherwise><cn cellml:units="per_s">0</cn></otherwise>
                </piecewise>
            </apply>
        </math>
    </component>
    <component name="logic_rate_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="k" public_interface="in" units="dimensionless"/>
        <variable name="a" public_interface="in" units="per_s"/>
        <variable name="b" public_interface="in" units="per_s"/>
        <variable initial_value="0" name="y" public_interface="out" units="dimensionless"/>
        <variable initial_value="0" name="z" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <!-- dy/dt = min(a, b) if (k > 2 or not(k == 2)) else 0: true for k = 1 -->
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply>
                <piecewise>
                    <piece>
                        <apply><min/><ci>a</ci><ci>b</ci></apply>
                        <apply><or/>
                            <apply><gt/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            <apply><not/>
                                <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            </apply>
                        </apply>
                    </piece>
                    <otherwise><cn cellml:units="per_s">0</cn></otherwise>
                </piecewise>
            </apply>
            <!-- dz/dt = a if (k > 2 or k == 2) else b: false for k = 1 -->
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply>
                <piecewise>
                    <piece>
                        <ci>a</ci>
                        <apply><or/>
                            <apply><gt/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                        </apply>
                    </piece>
                    <otherwise><ci>b</ci></otherwise>
                </piecewise>
            </apply>
        </math>
    </component>
</model>
"""

LOGIC_MODULES_CONFIG = """\
[
  {
    "vessel_type": "eq_rate",
    "BC_type": "nn",
    "module_format": "cellml",
    "module_file": "logic_test_modules.cellml",
    "module_type": "eq_rate_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["k", "dimensionless", "access", "constant"],
      ["x", "dimensionless", "access", "variable"]
    ]
  },
  {
    "vessel_type": "logic_rate",
    "BC_type": "nn",
    "module_format": "cellml",
    "module_file": "logic_test_modules.cellml",
    "module_type": "logic_rate_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["k", "dimensionless", "access", "constant"],
      ["a", "per_s", "access", "constant"],
      ["b", "per_s", "access", "constant"],
      ["y", "dimensionless", "access", "variable"],
      ["z", "dimensionless", "access", "variable"]
    ]
  }
]
"""


@pytest.mark.integration
@pytest.mark.parametrize('k, expected_x_rate', [(1.0, 1.0), (3.0, 0.0)])
def test_python_model_with_eq_or_not_min_runs(tmp_path, k, expected_x_rate):
    """#526: a piecewise with <eq/> (and <or/>, <not/>, <min/>) generates and runs as
    model_type python. It used to fail with ``NameError: name 'eq_func' is not defined``."""
    modules_dir = tmp_path / 'external_modules'
    modules_dir.mkdir()
    (modules_dir / 'logic_test_modules.cellml').write_text(LOGIC_MODULES_CELLML)
    (modules_dir / 'logic_test_modules_config.json').write_text(LOGIC_MODULES_CONFIG)

    generated_dir = _generate(
        tmp_path, 'logic_test',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        eqmod,nn,eq_rate,,
        logic,nn,logic_rate,,
        """,
        f"""\
        variable_name,units,value,data_reference
        k_eqmod,dimensionless,{k},test
        k_logic,dimensionless,{k},test
        a_logic,per_s,2.0,test
        b_logic,per_s,0.5,test
        """,
        model_type='python', external_modules_dir=str(modules_dir))

    source = (generated_dir / 'logic_test.py').read_text()
    for helper in ('eq_func', 'or_func', 'not_func', 'min('):
        assert helper in source

    t, res = _run(generated_dir / 'logic_test.py',
                  ['eqmod/x', 'logic/y', 'logic/z'], model_type='python')
    np.testing.assert_allclose(res['eqmod/x'], expected_x_rate * t, atol=1e-8)
    # k = 1: k > 2 or not(k == 2) -> min(a, b) = 0.5; k == 2 is false -> dz/dt = b.
    # k = 3: k > 2 is true           -> min(a, b) = 0.5; k > 2 is true   -> dz/dt = a.
    np.testing.assert_allclose(res['logic/y'], 0.5 * t, atol=1e-8)
    np.testing.assert_allclose(res['logic/z'], (0.5 if k == 1.0 else 2.0) * t, atol=1e-8)


def _helper_namespace(casadi_compat=False):
    namespace = {}
    exec('from math import *\nimport casadi as ca\n', namespace)
    lines = PythonGenerator._comparison_helper_lines(casadi_compat=casadi_compat)
    exec('\n'.join(lines), namespace)
    return namespace


def _libcellml_profile_namespace():
    """The helpers exactly as libCellML's Python profile writes them."""
    lc = pytest.importorskip('libcellml')
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
    namespace = {}
    exec('from math import *\n', namespace)
    for name in _profile_function_names():
        exec(getattr(profile, f'{name}FunctionString')(), namespace)
    return namespace


def _profile_function_names():
    return ['eq', 'neq', 'lt', 'leq', 'gt', 'geq', 'and', 'or', 'xor', 'not', 'min', 'max',
            'sec', 'csc', 'cot', 'sech', 'csch', 'coth',
            'asec', 'acsc', 'acot', 'asech', 'acsch', 'acoth']


def test_python_helpers_cover_the_libcellml_profile():
    """#526: every helper libCellML's Python profile can emit is defined and exported,
    under the name the profile calls it by."""
    lc = pytest.importorskip('libcellml')
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
    profile_names = set()
    for name in _profile_function_names():
        code = getattr(profile, f'{name}FunctionString')()
        profile_names.add(code.split('def ', 1)[1].split('(', 1)[0])
    assert profile_names == set(PythonGenerator.HELPER_FUNCTION_NAMES)

    for casadi_compat in (False, True):
        namespace = _helper_namespace(casadi_compat)
        for name in profile_names:
            assert callable(namespace.get(name)), (name, casadi_compat)
    aadc_code = '\n'.join(PythonGenerator._comparison_helper_lines(False, aadc_compat=True))
    for name in profile_names:
        assert f'def {name}(' in aadc_code


@pytest.mark.parametrize('casadi_compat', [False, True])
def test_python_helpers_match_libcellml_semantics(casadi_compat):
    """#526: the plain and CasADi helpers return what libCellML's own helpers return."""
    reference = _libcellml_profile_namespace()
    ours = _helper_namespace(casadi_compat)

    def value(v):
        return float(v)

    binary = ['eq_func', 'neq_func', 'lt_func', 'leq_func', 'gt_func', 'geq_func',
              'and_func', 'or_func', 'xor_func', 'min', 'max']
    for name in binary:
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.5, -1.0), (-3.0, 2.0)]:
            if casadi_compat and name == 'and_func' and (x < 0 or y < 0):
                continue  # the CasADi and_func treats only positive values as true
            assert value(ours[name](x, y)) == reference[name](x, y), (name, x, y)
    for x in (0.0, 1.0, -2.0):
        assert value(ours['not_func'](x)) == reference['not_func'](x)

    unary = {'sec': 0.4, 'csc': 0.4, 'cot': 0.4, 'sech': 0.4, 'csch': 0.4, 'coth': 0.4,
             'asec': 2.0, 'acsc': 2.0, 'acot': 2.0, 'asech': 0.5, 'acsch': 0.5, 'acoth': 2.0}
    for name, x in unary.items():
        assert math.isclose(value(ours[name](x)), reference[name](x), rel_tol=1e-12), name


def test_casadi_helpers_accept_symbolic_arguments():
    """#526: the CasADi variants build expressions from SX symbols rather than failing on
    Python truthiness."""
    ca = pytest.importorskip('casadi')
    ours = _helper_namespace(casadi_compat=True)
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    for name in ('eq_func', 'neq_func', 'or_func', 'xor_func', 'min', 'max'):
        f = ca.Function(name, [x, y], [ours[name](x, y)])
        assert float(f(1.0, 2.0)) == float(_helper_namespace()[name](1.0, 2.0))
    f = ca.Function('not', [x], [ours['not_func'](x)])
    assert float(f(0.0)) == 1.0
    for name in ('sec', 'acoth'):
        f = ca.Function(name, [x], [ours[name](x)])
        assert math.isclose(float(f(2.0)), _helper_namespace()[name](2.0))


# ---------------------------------------------------------------------------------------
# #525: an empty 'sum' multi-port is 0
# ---------------------------------------------------------------------------------------

@pytest.mark.integration
def test_empty_sum_multi_port_is_zero(tmp_path, capsys):
    """#525: a volume_sum with nothing connected generates, is mapped to its vessel, and is
    0, with a warning naming the vessel. It used to raise ``IndexError``."""
    vessel_array = _read_resource('3compartment_vessel_array.csv').rstrip('\n') + \
        '\nextra_sum, nn, volume_sum, , \n'
    generated_dir = _generate(tmp_path, 'empty_sum', vessel_array,
                              _read_resource('3compartment_parameters.csv'))
    warnings = [line for line in capsys.readouterr().out.splitlines() if 'WARNING' in line]
    assert any('extra_sum' in line for line in warnings), warnings

    model_text = (generated_dir / 'empty_sum.cellml').read_text()
    assert '<cn cellml:units="m3">0</cn>' in model_text
    # ... and the sum is still mapped to the vessel's port variable.
    assert 'variable_1="q_extra_sum_sum" variable_2="q"' in model_text

    # Myokit merges connected variables, so read the sums where they are computed.
    _, res = _run(generated_dir / 'empty_sum.cellml',
                  ['sum_blood_volume/q_extra_sum_sum', 'sum_blood_volume/q_volume_sum_sum'],
                  sim_time=0.1)
    np.testing.assert_array_equal(res['sum_blood_volume/q_extra_sum_sum'], 0.0)
    # The populated sum is untouched.
    assert np.all(res['sum_blood_volume/q_volume_sum_sum'] > 0.0)
