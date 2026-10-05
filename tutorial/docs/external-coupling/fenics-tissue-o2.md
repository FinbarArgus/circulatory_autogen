# Tissue O2 and a microvasculature model, in FEniCS

Two capillaries deliver O2 to a box of tissue, where it diffuses and is consumed. The capillaries
are 0D CellML modules in generated C++. The tissue is a FEniCSx model, and the same tissue can
also be a finite-volume grid of CellML cells, so the two can be compared.

The files are in the module library
[circulatory-autogen-modules](https://github.com/physiomelinks/circulatory-autogen-modules):

| Path | What it is |
|---|---|
| `modules/transport/tissue_diffusion_FEniCS/versions/box_v01/` | the FEniCS module: config, `tissue_diffusion_FEniCS_box_v01_model.py`, instances `default` (tissue O2) and `NE_extracellular` |
| `system_models/coupled/microvasc_O2_FEniCS/` | capillaries + FEniCS tissue |
| `system_models/coupled/microvasc_O2_FV/` | the same capillaries + a finite-volume grid of `tissue_diffusion_volume` cells |
| `tools/build_coupled_examples.py` | builds both system models |
| `tests/test_coupled_systems.py` | runs and compares them |

## The model

**Capillaries.** Each one is a haemodynamic `capillary` (`pp`) feeding a `GE_capillary`, which
computes O2 exchange with the tissue around it:

```
flux_O2_c = 2 pi r L perm_O2 (ub_O2_c - ub_O2_t)        [mol/s, into the tissue]
```

`ub_O2_t`, the tissue O2 at the capillary, is a boundary condition of `GE_capillary`, given
through its `capillary_to_flux_port [ub_O2_t, flux_O2_c]`.

**Tissue.** One solute in a box, with no flux through the faces:

```
dC/dt = sigma_diff lap(C) + sum_k chi_k J_k / V_k + M C / (C + C50)
```

Each connected capillary owns one cell of a `grid_nx × grid_ny × grid_nz` block grid of the box:
neighbour `k` of `N` sits at x-cell `floor((k + 1/2) grid_nx / N)`, in the middle of y and z. It
sends its flux `J_s` into that cell as a uniform source, and receives the cell's mean
concentration `C_t`. These are the variables of the module's `capillary_to_flux_port`, the
port the finite-volume CellML cell `tissue_diffusion_volume` has too, so a model swaps one for
the other by changing a row.

## The module config

```json
{
  "module_type": "tissue_diffusion_FEniCS", "module_subtype": "box_v01",
  "module_format": "external_api",
  "entrance_ports": [{"port_type": "capillary_to_flux_port", "variables": ["C_t", "J_s"]}],
  "variables_and_units": [
    ["C_t", "millimolar", "access", "variable"],
    ["J_s", "mol_per_s", "access", "variable"],
    ["Lx", "metre", "access", "constant"], "...",
    ["sigma_diff", "m2_per_s", "access", "global_constant"]
  ],
  "api": {
    "role": "provider", "transport": "python",
    "python": {"file": "tissue_diffusion_FEniCS_box_v01_model.py", "class": "DiffusionBox"},
    "coupling_dt": 0.01
  }
}
```

The directions follow from `GE_capillary`:
- `ub_O2_t` is a boundary condition, so the FEniCS model sets `C_t`;
- `flux_O2_c` is computed, so it receives `J_s`.

One row connects to both capillaries (`inp_instances: [capillary_GE_0, capillary_GE_1]`), so each
variable is an array of two values.

## The FEniCS model

`DiffusionBox` (dolfinx 0.8 or 0.9) builds everything once: the mesh, the function space, the
region indicators and the forms. Each step then writes the source into a DG0 function, solves
one linear system, and returns the region means:

```python
def step(self, t, dt, inputs):
    J = inputs['J_s']                                   # one flux per capillary [mol/s]
    self.src.x.array[:] = 0.0
    for chi, Jk, Vk in zip(self.chi, J, self.volumes):
        self.src.x.array[:] += chi.x.array * (Jk / Vk)  # [mM/s] in the capillary's cell
    ...                                                 # assemble b, solve with the cached LU
    return {'C_t': self.region_means()}                 # summed over MPI ranks
```

- **Time stepping:** theta method (`theta` 0.5, Crank–Nicolson), with the source and the
  consumption taken at the start of the step. The matrix is factorised once per step size.
- **Two discretisations** (`fv_scheme`):
  - `0`: Q1 finite elements, `refine` elements per grid cell and direction;
  - `1`: one DG0 value per grid cell with the two-point face flux `sigma_diff A_f (C_a − C_b) / |x_a − x_b|`.

  With `1` the discrete equations are those of the CellML grid of `tissue_diffusion_volume`
  cells joined by `tissue_diffusion_face` faces, which makes the coupling checkable exactly.
- **Scaling:** lengths are scaled by the box's largest side before assembly. In metres the
  matrix entries are about 1e-15, below PETSc's LU zero-pivot tolerance.

## Running it

```bash
cd circulatory-autogen-modules
cuflynx-generate ...   # or, as the test does: tests/test_coupled_systems.py::generate_cpp
cuflynx-couple <generated>/microvasc_O2_FEniCS
```

## FEniCS against the CellML grid

`tests/test_coupled_systems.py` runs:
- `microvasc_O2_FV` (all CellML, C++ through `main0d`);
- `microvasc_O2_FEniCS` (C++ 0D + FEniCS through `cuflynx-couple`), with each FEniCS discretisation.

It compares the tissue O2 at the two capillaries over the run and records the times in
`system_models/coupled/microvasc_O2_FEniCS/results/coupled_comparison.json`.

RESULTS_O2
