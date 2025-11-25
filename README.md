# Deep-Potential Long-Range Quantum

This is the implementation of Deep-Potential Long-Range Quantum (DPLR-q), where an excess electron is present in the system and explicitly modeled, while the long-range Coulomb interaction and short-range interaction are treated by a Deep Potential model. The code is mostly only for reproducing the results in the associated paper (...link to be added later...). It includes system-specific snippets, and modifications may be needed to apply to other systems. For the general-purpose DP/DPLR in jax, please refer to [deepmd-jax](https://github.com/SparkyTruck/deepmd-jax).

## Installation
Note: You need to first have **CUDA 12** installed for GPU support.
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
    grid_size = 48,
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
The `scripts` directory contains simulation and post-processing scripts for the specific system e⁻(aq) + H⁺(aq) → H·(aq), which are less organized and intended mainly for ad-hoc analysis.