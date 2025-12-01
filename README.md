# Deep-Potential Long-Range Quantum (DPLR-q)

This repository contains the research implementation of **Deep-Potential Long-Range Quantum (DPLR-q)**, an electron-aware machine-learning force field in which an *excess electron* is treated explicitly at the wave-function level, while long-range Coulomb interactions and short-range many-body interactions are modeled with a Deep Potential network.

DPLR-q was developed and used in the work:

> Ruiqi Gao, Pinchen Xie, and Roberto Car,  
> **“A Machine Learning Model for the Chemistry of a Solvated Electron”**,  
> arXiv:2511.22642 — <https://arxiv.org/abs/2511.22642>

This code is primarily intended to **reproduce the results** reported in the associated paper, where there are **system-specific scripts and settings** tailored to the solvated-electron / hydronium system studied in the paper. Adapting DPLR-q to other systems could require modifications.

For a **general-purpose** implementation of Deep Potential / DPLR models in JAX, please refer instead to:

- [`deepmd-jax`](https://github.com/SparkyTruck/deepmd-jax)


## Installation
Note: The code is currently written to run on a single GPU.
```
git clone https://github.com/SparkyTruck/dplr-q.git
cd dplr-q
pip install -e .
```

## Training a model

The dataset from DFT is available on [Zenodo](https://doi.org/10.5281/zenodo.17684097).
```python
from deepmd_jax.train import train
```
First, train a Wannier centroid model:
```python
train(
        model_type='atomic',
        rcut=6,
        mp=True,
        save_path='wannier_e_and_H.pkl',
        train_data_path=list_of_train_data_paths,
        atomic_sel=[0],
        step=100000,
    )
```
Next, train the short-range model. The saved model includes both the Wannier centroid model and the short-range model.
```python
train(
        model_type='dplr_q1', # Deep-Potential Long-Range Quantum with one excess electron
        rcut=6,
        mp=True,
        save_path='model_e_and_H.pkl',
        train_data_path=list_of_train_data_paths,
        step=1000000,
        dplr_wannier_model_path = 'wannier_e_and_H.pkl',
        dplr_q_atoms = [6, 1], # O and H
        dplr_q_wc = [-8],      # Wannier centroid charge for each oxygen
        dplr_beta = 0.6,       # smoothing for long-range interaction
        dplr_q1_beta = 2.0,    # smoothing for the embedding potential of the excess electron
    )
```

## Simulation

```python
from deepmd_jax.md import Simulation
import numpy as np
initial_position = np.load('water_64_with_e_and_H.npy')
sim = Simulation(
    model_path = 'model_e_and_H.pkl',
    grid_size = 48, # recommends around box*4 with a number with small prime factors
    box=12.4171,
    type_idx=np.array([0,1,1] * 64 + [1]),
    mass=[15.9994, 1.0078],
    routine='NVT',
    dt=0.5,
    tau_t=100,
    initial_position=initial_position,
    temperature=350,
    report_interval=10,
)
trajectory = sim.run(2000)
 # predicted electron centers are recorded every report_interval
assert trajectory['center'].shape == trajectory['position'][::10].shape
```

### Miscellaneous
The `scripts` directory contains simulation and post-processing scripts for the specific system e⁻(aq) + H⁺(aq) → H·(aq), which are less organized and intended mainly for ad-hoc analysis.

---

# DWIR

This repository also contains a research implementation of **Deep Wannier Iterative Refinement (DWIR)** for a single electron, a method to iteratively refine the prediction of Wannier centers of an electron not uniquely associated with any specific atom.

Example code to train a DWIR model for the solavated electron system:
```python
from deepmd_jax.train import train
train(
      model_type='atomic_iter',     
      rcut=6.0,        
      mp=True,
      save_path=f'trained_models/dwir_model.pkl',
      train_data_path=list_of_train_data_paths,
      step=50000,
      gamma_iter=2,   # loss coefficient in iterative refinement
      n_iter=4,       # number of refinement iterations during training
      perturb_iter=1, # purturbation strength (Angstrom) for initial guess
      batch_size=32,
      lr=0.01,
)
```

The DWIR model does not play a part in molecular dynamics simulations, but it can be used as a Wannier center predictions in post-processing analysis of standard Deep Potential simulations. For DPLR-q, this is generally not necessary as it directly outputs the centers in the trajectory. However, DWIR can be somewhat more accurate in terms of just the centers.

To inference the Wannier center for a given trajectory, start with a not-too-bad initial guess `init_wc` of shape `(1, 3)` for the first frame, and use the standard `evaluate` function:
```python
from deepmd_jax.train import evaluate
wannier_predictions = evaluate('trained_models/dwir_model.pkl', trajectory['position'], trajectory['box'], type_idx, init_wc=init_wc, n_iter=4)
```