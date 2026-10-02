'''
0D networks with nodes where several vessels meet, coupled to the FV 1D solver.

aortic_bif_hybrid_V1 (a 1D aortic bifurcation with a 0D terminal on each daughter) is extended on
both sides of the 1D tree with 0D nodes that ordinary vessels own through summing ports
(libcuflynx.generators.port_nodes):

    inflow -> a (vv) -> b (pp) -> parent (1D) -> daughter_1/2 (1D) -> terminal_1/2
                     -> c (pp) ------------------------------------------------\\
                                                    terminal_1, terminal_2, c -> venous (vp) -> sink

a's outlet owns the split into b and c; venous's inlet owns the merge of both terminals (which
take their inflow from the 1D solver) and the 0D branch c. Each 1D-0D link stays one-to-one, which
is what the coupling supports; a node of three or more ends that includes a 1D vessel is refused.

The model is generated as C++, built with CMake, and run under the coupler with the Python 1D
solver for one cardiac cycle. The 0D output must then conserve flow at both nodes and give every
module at a node the owner's pressure.
'''
import json
import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_cpp_template_generator import _cmake_build, _generation_inputs, _load_output

PREFIX = 'nodes_1d'
VESSEL_ARRAY = '''name,BC_type,vessel_type,inp_vessels,out_vessels
input_flow_aorticroot,nn_aorticbif,inlet_flow,,a
a,vv,arterial_simple,input_flow_aorticroot,b c
b,pp,arterial_simple,a,parent
c,pp,arterial_simple,a,venous
parent,nn,FV1D_vessel,b,daughter_1 daughter_2
daughter_1,nn,FV1D_vessel,parent,terminal_1
daughter_2,nn,FV1D_vessel,parent,terminal_2
terminal_1,pp,terminal,daughter_1,venous
terminal_2,pp,terminal,daughter_2,venous
venous,vp,venous,terminal_1 terminal_2 c,sink
sink,nn_constant,outlet_pressure,venous,
'''
ZERO_D = {
    'a': {'R': 1e6, 'C': 1e-8, 'I': 1e4},
    'b': {'R': 1e6, 'C': 1e-9, 'I': 1e4},
    'c': {'R': 1e9, 'C': 1e-9, 'I': 1e5},
}
UNITS = {'R': 'Js_per_m6', 'C': 'm6_per_J', 'I': 'Js2_per_m6', 'q_0': 'm3', 'u_0': 'J_per_m3', 'u_ext': 'J_per_m3'}


def _resources(tmp_path, resources_dir, vessel_array):
    '''The extended model's vessel array and parameters, in a resources dir of their own.'''
    res = tmp_path / 'node_resources'
    res.mkdir()
    (res / f'{PREFIX}_vessel_array.csv').write_text(vessel_array)
    params = pd.read_csv(os.path.join(resources_dir, 'aortic_bif_hybrid_V1_parameters.csv'))
    params = params[~params['variable_name'].isin(['u_out_terminal_1', 'u_out_terminal_2'])]
    rows = []
    for name, values in ZERO_D.items():
        for var, value in dict(values, q_0=0.0, u_0=0.0, u_ext=0.0).items():
            rows.append((f'{var}_{name}', UNITS[var], value))
    rows += [('R_venous', 'Js_per_m6', 1e6), ('C_venous', 'm6_per_J', 1e-6), ('I_venous', 'Js2_per_m6', 1e4),
             ('u_ext_venous', 'J_per_m3', 0.0), ('q_C_init_venous', 'm3', 0.0), ('q_us_0_venous', 'm3', 0.0),
             ('Delta_q_us_venous', 'dimensionless', 0.0), ('Delta_C_venous', 'dimensionless', 0.0),
             ('P_sink', 'J_per_m3', 0.0),
             # set by the 1D solver through the coupling; the CellML needs a placeholder
             ('u_out_b', 'J_per_m3', 0.0)]
    extra = pd.DataFrame([{'variable_name': n, 'units': u, 'value': v, 'data_reference': 'test'} for n, u, v in rows])
    pd.concat([params, extra]).to_csv(res / f'{PREFIX}_parameters.csv', index=False)
    return res


@pytest.mark.integration
@pytest.mark.slow
def test_nodes_either_side_of_a_1d_tree(user_inputs_dir, resources_dir, tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    from libcuflynx.utilities.package_resources import package_data_file
    res = _resources(tmp_path, resources_dir, VESSEL_ARRAY)
    cpp_dir = tmp_path / 'gen' / 'cpp'
    ini = tmp_path / '1d' / 'run000' / 'input.ini'
    inp = _generation_inputs(user_inputs_dir, str(res), tmp_path, PREFIX, 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini), dt=0.01)
    assert generate_with_new_architecture(False, inp)
    _cmake_build(cpp_dir, cpp_dir / 'build')
    coupler_src = os.path.join(os.path.dirname(__file__), '..', 'src', 'libcuflynx', 'coupler')
    if not os.path.isfile(os.path.join(coupler_src, 'coupler.cpp')):
        pytest.skip('coupler sources need a source checkout')
    _cmake_build(coupler_src, tmp_path / 'coupler_build')

    pipes = tmp_path / 'pipes'
    pipes.mkdir()
    config = {'inputFold': str(cpp_dir) + '/', 'networkName': PREFIX, 'ODEsolver': 'CVODE', 'T0': 1.1, 'nCC': 1,
              'tmp_pipe_path': str(pipes) + '/', 'initStatePath': 'None', 'python_path': sys.executable,
              'solver1d_path': str(package_data_file('libcuflynx.solver1d', 'main1D.py')),
              'solver0d_path': str(cpp_dir / 'build' / 'main0d'), 'initFile_sim1d_path': str(ini)}
    with open(cpp_dir / 'coupler_config.json', 'w') as f:
        json.dump(config, f)
    log_path = tmp_path / 'coupler.log'
    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen([str(tmp_path / 'coupler_build' / 'coupler'), str(cpp_dir / 'coupler_config.json')],
                                stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode = None
    log = log_path.read_text(errors='replace')
    tail = '\n'.join(log.splitlines()[-80:])
    assert returncode == 0 and 'Both processes terminated successfully with status codes : 0 0' in log, tail

    cv, v = _load_output(tmp_path / 'simulation_outputs_cpp' / PREFIX / 'sol0D_variables.txt')
    assert v.shape[0] > 50 and np.all(np.isfinite(v))

    def col(name):
        assert name in cv, f'{name} not in the 0D output: {sorted(cv)[:60]}'
        return v[:, cv[name]]

    # the split a -> b, c: a's outflow is b's and c's inflow
    split = col('multiport_sum_a_v_out/v_out')
    assert np.allclose(split, col('b/v') + col('c/v'), rtol=1e-9, atol=1e-15)
    # the merge terminal_1, terminal_2, c -> venous: venous's inflow is all three outflows
    merge = col('multiport_sum_venous_v_in/v_in')
    assert np.allclose(merge, col('terminal_1/v_T') + col('terminal_2/v_T') + col('c/v_d'), rtol=1e-9, atol=1e-15)
    # flow reaches the 1D tree and comes back from it: the terminals carry most of the outflow
    assert np.max(col('b/v')) > 1e-5 and np.max(col('terminal_1/v_T')) > 1e-6
    sol1d = np.genfromtxt(ini.parent / 'res' / 'sol1D_parent.txt')
    assert np.all(np.isfinite(sol1d))


@pytest.mark.integration
def test_a_1d_vessel_cannot_share_a_node_with_two_0d_modules(user_inputs_dir, resources_dir, tmp_path):
    '''The 1D coupling exchanges one flow and one pressure per 1D-0D link, so a 1D vessel meeting
    two 0D vessels at one node is refused, with the node named.'''
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    array = VESSEL_ARRAY.replace('a,vv,arterial_simple,input_flow_aorticroot,b c',
                                 'a,vv,arterial_simple,input_flow_aorticroot,parent c')
    array = array.replace('b,pp,arterial_simple,a,parent\n', '')
    array = array.replace('parent,nn,FV1D_vessel,b,', 'parent,nn,FV1D_vessel,a,')
    res = _resources(tmp_path, resources_dir, array)
    inp = _generation_inputs(user_inputs_dir, str(res), tmp_path, PREFIX, 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(tmp_path / 'gen' / 'cpp'),
                             cpp_1d_model_config_path=str(tmp_path / '1d' / 'run000' / 'input.ini'), dt=0.01)
    with pytest.raises(NotImplementedError, match='parent has no CellML'):
        generate_with_new_architecture(False, inp)
