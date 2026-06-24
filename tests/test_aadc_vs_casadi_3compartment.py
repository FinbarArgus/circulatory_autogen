"""
AADC vs CasADi comparison on the stiff 3compartment model.

Compares the two automatic-differentiation backends on three axes, all using a BDF
stiff solve where applicable:
  1. Forward-solve accuracy vs the Myokit CVODE reference.
  2. Gradient accuracy vs finite differences (FD).
  3. Wall-clock speed of the forward solve.

Both backends generate from the same CellML, so their state names are identical
(``component/variable``); the Myokit reference uses ``component_module.variable`` and is
matched component-wise.

Notes / current status (see PR #251):
  - CasADi's ``method='bdf'`` (symbolic implicit BDF2 with a rootfinder per step) matches
    CVODE on this stiff model, and its own gradient matches FD — the BDF graph is fully
    differentiable, so AD works directly on bdf (no fallback to semi_implicit_euler).
    These tests pass.
  - AADC's ``method='bdf'`` and tape gradient are **license-gated** (Matlogica). Without a
    license they raise ``RuntimeError: AADC License check failed``, so the AADC-side tests
    fail here until a license is provisioned (locally and on CI). They are intentionally
    left failing rather than skipped.
"""
import os
import sys
import time

import numpy as np
import pytest

_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from solver_wrappers import get_simulation_helper
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from utilities.utility_funcs import get_default_inp_data_dict

_DT = 0.001
_SIM_TIME = 1.0


@pytest.fixture(scope="module")
def models_3compartment(tmp_path_factory):
    """Generate the casadi_python and aadc_python 3compartment models once for the module."""
    base = str(tmp_path_factory.mktemp("aadc_vs_casadi_3c"))
    resources = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

    def _gen(model_type, solver, solver_info, subdir):
        d = os.path.join(base, subdir)
        os.makedirs(d, exist_ok=True)
        inp = get_default_inp_data_dict("3compartment", "3compartment_parameters.csv", resources)
        inp.update({
            "model_type": model_type,
            "generated_models_dir": d,
            "solver": solver,
            "solver_info": solver_info,
        })
        assert generate_with_new_architecture(False, inp), f"generation failed for {model_type}"
        return {
            "py": os.path.join(d, "3compartment", "3compartment.py"),
            "cellml": os.path.join(d, "3compartment", "3compartment.cellml"),
        }

    return {
        "casadi": _gen("casadi_python", "casadi_integrator", {"method": "bdf"}, "ca"),
        "aadc": _gen("aadc_python", "aadc_semi_implicit", {"method": "bdf"}, "aadc"),
    }


# ---- helpers ----
def _norm_state_key(name):
    """Normalise a state name to (component, variable) for cross-backend matching.

    Myokit:  component_module.variable   CasADi/AADC/Python: component/variable
    """
    if '/' in name:
        comp, var = name.split('/', 1)
    elif '.' in name:
        comp, var = name.split('.', 1)
    else:
        return (None, name.lower())
    return (comp.replace('_module', '').lower(), var.lower())


def _max_state_rel_l2_vs_cvode(cvode_helper, other_helper):
    """Max over matched ODE states of the whole-trajectory relative-L2 error (%) between
    a CVODE_myokit reference helper and another backend's helper."""
    cv_names = cvode_helper.get_all_variable_names()
    cv_res = cvode_helper.get_all_results(flatten=False)
    cv_map = {_norm_state_key(nm): np.asarray(cv_res[i][0]).flatten() for i, nm in enumerate(cv_names)}

    worst, worst_var, compared = 0.0, None, 0
    for sname in other_helper.state_name_to_idx:
        key = _norm_state_key(sname)
        if key not in cv_map:
            continue
        a = np.asarray(other_helper.get_results([[sname]], flatten=True)[0]).flatten()
        b = cv_map[key]
        L = min(len(a), len(b))
        if L < 2:
            continue
        a, b = a[:L], b[:L]
        denom = np.linalg.norm(b)
        if denom < 1e-12:
            continue
        err = np.linalg.norm(a - b) / denom * 100.0
        compared += 1
        if err > worst:
            worst, worst_var = err, sname
    return worst, worst_var, compared


def _cvode_reference(cellml_path):
    ref = get_simulation_helper(
        model_path=cellml_path, solver='CVODE_myokit', model_type='cellml_only',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0,
        solver_info={'MaximumStep': 1e-4, 'rtol': 1e-8, 'atol': 1e-10},
    )
    ref.run()
    return ref


# ---- 1. Forward-solve accuracy ----
@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_forward_casadi_bdf_vs_cvode(models_3compartment):
    """CasADi BDF (symbolic implicit BDF2 + rootfinder) should match the CVODE reference."""
    paths = models_3compartment["casadi"]
    ref = _cvode_reference(paths["cellml"])

    sim = get_simulation_helper(
        model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    sim.run()

    max_pct, worst_var, compared = _max_state_rel_l2_vs_cvode(ref, sim)
    print(f"\nCasADi BDF vs CVODE: max rel-L2 {max_pct:.3f}% on {worst_var} ({compared} states)")
    assert compared > 0, "No states matched between CVODE and CasADi"
    assert max_pct < 2.0, f"CasADi BDF deviates from CVODE by {max_pct:.3f}% on {worst_var}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_forward_aadc_bdf_vs_cvode(models_3compartment):
    """AADC BDF (scipy BDF + AADC Jacobian) should match the CVODE reference.

    License-gated: raises ``AADC License check failed`` without a Matlogica license.
    """
    pytest.importorskip("aadc")
    paths = models_3compartment["aadc"]
    ref = _cvode_reference(paths["cellml"])

    sim = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    sim.run()  # license check happens here (AADC VectorFunctionWithJacobian)

    max_pct, worst_var, compared = _max_state_rel_l2_vs_cvode(ref, sim)
    print(f"\nAADC BDF vs CVODE: max rel-L2 {max_pct:.3f}% on {worst_var} ({compared} states)")
    assert compared > 0, "No states matched between CVODE and AADC"
    assert max_pct < 5.0, f"AADC BDF deviates from CVODE by {max_pct:.3f}% on {worst_var}"


# ---- 2. Gradient accuracy vs FD ----
def _pick_param_and_state(helper):
    pname = next((n for n in helper.var_name_to_idx if 'e_lv_a' in n.lower()),
                 list(helper.var_name_to_idx.keys())[0])
    sname = next((n for n in helper.state_name_to_idx if 'q_lv' in n.lower()),
                 list(helper.state_name_to_idx.keys())[0])
    return pname, sname


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_casadi(models_3compartment):
    """CasADi symbolic gradient (semi_implicit_euler differentiable path) vs FD.

    cost = (final value of a state)^2; d(cost)/d(param) by CasADi AD vs central FD.
    """
    import casadi as ca
    paths = models_3compartment["casadi"]

    def _helper():
        return get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'semi_implicit_euler'},
        )

    h = _helper()
    pname, sname = _pick_param_and_state(h)
    p0 = float(h.get_init_param_vals([[pname]])[0])
    sidx = h.state_name_to_idx[sname]

    # AD: differentiate cost through the symbolic semi-implicit Euler trajectory.
    h._create_param_subset([[pname]], [p0])
    h.run()
    cost_symb = h.state_traj_symb[sidx, -1] ** 2
    jac = ca.gradient(cost_symb, h.variables_symb_subset)
    jac_func = ca.Function('jac_cost', [h.states_symb, h.variables_symb], [jac])
    g_ad = float(np.array(jac_func(h.states, h.variables)).flatten()[0])

    # FD on the same scheme.
    def cost_at(pv):
        hh = _helper()
        hh.set_param_vals([[pname]], [[pv]])
        hh.run()
        v = np.asarray(hh.get_results([[sname]], flatten=True)[0]).flatten()[-1]
        return float(v) ** 2

    step = abs(p0) * 1e-6 if p0 != 0 else 1e-6
    g_fd = (cost_at(p0 + step) - cost_at(p0 - step)) / (2 * step)

    print(f"\nCasADi gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"CasADi AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-12, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_casadi_bdf(models_3compartment):
    """CasADi symbolic gradient through the implicit **BDF** path vs FD.

    The BDF solver is a symbolic CasADi graph (rootfinder per step), so it supports
    AD: ``state_traj_symb`` is populated and ``d(cost)/d(param)`` by CasADi AD must
    match central finite differences. Same cost = (final value of a state)^2 as the
    semi-implicit gradient test, but on ``method='bdf'`` — proving the former scipy
    BDF guard (do_ad rejected) is gone and AD now works on bdf.
    """
    import casadi as ca
    paths = models_3compartment["casadi"]

    def _helper():
        return get_simulation_helper(
            model_path=paths["py"], solver='casadi_integrator', model_type='casadi_python',
            dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'bdf'},
        )

    h = _helper()
    pname, sname = _pick_param_and_state(h)
    p0 = float(h.get_init_param_vals([[pname]])[0])
    sidx = h.state_name_to_idx[sname]

    # AD: differentiate cost through the symbolic implicit-BDF trajectory.
    h._create_param_subset([[pname]], [p0])
    assert h._do_ad is True
    h.run()
    assert h.state_traj_symb is not None  # symbolic graph exists -> AD is possible
    cost_symb = h.state_traj_symb[sidx, -1] ** 2
    jac = ca.gradient(cost_symb, h.variables_symb_subset)
    jac_func = ca.Function('jac_cost', [h.states_symb, h.variables_symb], [jac])
    g_ad = float(np.array(jac_func(h.states, h.variables)).flatten()[0])

    # FD on the same scheme.
    def cost_at(pv):
        hh = _helper()
        hh.set_param_vals([[pname]], [[pv]])
        hh.run()
        v = np.asarray(hh.get_results([[sname]], flatten=True)[0]).flatten()[-1]
        return float(v) ** 2

    step = abs(p0) * 1e-6 if p0 != 0 else 1e-6
    g_fd = (cost_at(p0 + step) - cost_at(p0 - step)) / (2 * step)

    print(f"\nCasADi BDF gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"CasADi BDF AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-12, f"FD≈0 but AD={g_ad:.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_gradient_vs_fd_aadc(models_3compartment):
    """AADC tape gradient vs FD on the same tape.

    License-gated: raises ``AADC License check failed`` without a Matlogica license.
    """
    aadc = pytest.importorskip("aadc")
    paths = models_3compartment["aadc"]

    sim = get_simulation_helper(
        model_path=paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=0.01, sim_time=0.3, pre_time=0.0, solver_info={'method': 'semi_implicit'},
    )
    sim.run()

    vi = sim.model.VARIABLE_INFO
    pidx = next(i for i, info in enumerate(vi) if 'e_lv_a' in info['name'].lower())
    sidx = next(i for i, info in enumerate(sim.model.STATE_INFO) if 'q_lv' in info['name'].lower())
    sim._ad_param_names = [vi[pidx]['name']]
    sim._ad_param_var_indices = [pidx]

    def cost_fn(st, p):
        return st[sidx] * st[sidx]

    g_ad = sim.compute_gradient_tape(cost_fn)[0]  # license check happens here

    pv = float(sim._numeric_variables_all[pidx])
    h = abs(pv) * 1e-5 if pv != 0 else 1e-5
    workers, funcs, rc, ap = sim._aad_workers, sim._tape_funcs, sim._tape_r_cost, sim._tape_a_p
    cp = float(np.asarray(aadc.evaluate(funcs, {rc: []}, {ap[0]: pv + h}, workers)[0][rc]).flat[0])
    cm = float(np.asarray(aadc.evaluate(funcs, {rc: []}, {ap[0]: pv - h}, workers)[0][rc]).flat[0])
    g_fd = (cp - cm) / (2 * h)

    print(f"\nAADC gradient vs FD: AD={g_ad:.6e} FD={g_fd:.6e}")
    if abs(g_fd) > 1e-30:
        ratio = g_ad / g_fd
        assert abs(ratio - 1.0) < 0.01, f"AADC AD/FD ratio = {ratio:.6f} (expected ~1.0)"
    else:
        assert abs(g_ad) < 1e-10, f"FD≈0 but AD={g_ad:.6e}"


# ---- 3. Speed ----
@pytest.mark.integration
@pytest.mark.slow
def test_3compartment_speed_casadi_vs_aadc_bdf(models_3compartment):
    """Wall-clock comparison of the BDF forward solve: CasADi vs AADC.

    Both are timed and printed. The AADC BDF path is license-gated, so this test fails
    on the AADC side until a Matlogica license is available; the CasADi timing is still
    printed first.
    """
    pytest.importorskip("aadc")

    ca_paths = models_3compartment["casadi"]
    aadc_paths = models_3compartment["aadc"]

    sim_ca = get_simulation_helper(
        model_path=ca_paths["py"], solver='casadi_integrator', model_type='casadi_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    t0 = time.perf_counter()
    sim_ca.run()
    t_casadi = time.perf_counter() - t0
    print(f"\nCasADi BDF forward solve: {t_casadi:.3f}s")
    assert np.all(np.isfinite(sim_ca.state_traj_dm[:, -1])), "CasADi BDF produced non-finite states"

    sim_aadc = get_simulation_helper(
        model_path=aadc_paths["py"], solver='aadc_semi_implicit', model_type='aadc_python',
        dt=_DT, sim_time=_SIM_TIME, pre_time=0.0, solver_info={'method': 'bdf'},
    )
    t0 = time.perf_counter()
    sim_aadc.run()  # license check happens here
    t_aadc = time.perf_counter() - t0
    print(f"AADC BDF forward solve: {t_aadc:.3f}s  (CasADi/AADC = {t_casadi / t_aadc:.2f}x)")
    assert np.all(np.isfinite(sim_aadc.state_traj[:, -1])), "AADC BDF produced non-finite states"
