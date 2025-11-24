# %env XLA_PYTHON_CLIENT_PREALLOCATE=false
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import vmap
box = 12.4171
from jax import jit
from functools import partial

### all for a system of water + e- + H+, type_idx = [0,1,1]*N + [1]

def analyze_traj(coords, centers=None, box=None, plot=True, check_bond=True):
    # input a batch of coordinates, shape (nframes, N, 3), center (nframes, 3)
    N = coords.shape[1] // 3
    if box is None:
        box = [12.4171, 15.6446, 19.711, 24.8342][int(np.log2(N))-6]
    O = jnp.array(coords[:, ::3][:, :-1])
    H = jnp.concatenate([coords[:,
                                1::3], coords[:,2::3], coords[:,-1:]], axis=1)
    distOH = jnp.linalg.norm((O[:,:,None] - H[:,None] + box/2) % box - box/2, axis=-1)
    H2_idx = jnp.argsort(distOH, axis=-1)[:,:,:2].reshape(-1,2*N)
    # Hyd is the one H left after removing the two H closest to O
    H_is_bonded = vmap(jnp.isin, in_axes=(None, 0))(jnp.arange(2*N+1), H2_idx)
    valid = True
    if check_bond:
        try:
            assert (H_is_bonded.sum(-1) == 2*N).all()
        except AssertionError:
            breakage_idx = jnp.where(H_is_bonded.sum(-1) < 2*N)[0]
            print('Warning: Bond breakage detected at ', breakage_idx)
            valid = False
    else:
        valid = (H_is_bonded.sum(-1) == 2*N).all()
    dOHsortO = jnp.sort(distOH, axis=1)
    dHO12 = jnp.sort(((dOHsortO[:, 1] - dOHsortO[:, 0]) * H_is_bonded), axis=-1)[:,1]
    dOH_bonded = jnp.sort(distOH, axis=-1)[:,:,:2]
    dOHmax = dOH_bonded.max((1,2))
    if check_bond:
        print('OH-bond:', dOH_bonded.min(), dOH_bonded.max())
    Hyd_idx = H_is_bonded.argsort(axis=-1)[:,0]
    Hyd = jnp.take_along_axis(H, Hyd_idx[:,None,None], axis=1)[:,0]
    dHydO = jnp.linalg.norm((O - Hyd[:,None] + box/2) % box - box/2, axis=-1)
    # O_Hyd is the O closest to Hyd
    O_Hyd_idx = dHydO.argmin(-1)
    dOHyd = dHydO.min(-1)
    dOHyd_2nd = jnp.sort(dHydO, axis=-1)[:,1]
    O_Hyd = jnp.take_along_axis(O, O_Hyd_idx[:,None,None], axis=1)[:,0]
    dHydC = None
    dOC = None
    dHCmin = None
    dOHydHmin = None
    if centers is not None:
        dHydC = jnp.linalg.norm((centers - Hyd + box/2) % box - box/2, axis=-1)
        dOC = jnp.linalg.norm((centers - O_Hyd + box/2) % box - box/2, axis=-1)
        dHCmin = jnp.linalg.norm((H - centers[:,None] + box/2) % box - box/2, axis=-1).min(-1)
        Hmin_idx = jnp.linalg.norm((H - centers[:,None] + box/2) % box - box/2, axis=-1).argmin(-1)
        Hmin = jnp.take_along_axis(H, Hmin_idx[:,None,None], axis=1)[:,0]
        dOHydHmin = jnp.linalg.norm((O_Hyd - Hmin + box/2) % box - box/2, axis=-1)
    if plot:
        plt.figure(figsize=(10, 8))
        markersize = 2
        plt.plot(dOHyd, '.', markersize=markersize, label='O-Hyd', alpha=0.5) # distance between Hyd and the closest O
        plt.plot(dOHyd_2nd, '.', markersize=markersize, label='O-Hyd2', alpha=0.5) # distance between Hyd and the second closest O
        if centers is not None:
            plt.plot(dHydC, '.', markersize=markersize, label='Hyd-C') # distance between Hyd and the electron center
            plt.plot(dOC, '.', markersize=markersize, label='O-C') # distance between O_Hyd and the electron center
            plt.plot(dHCmin, '.', markersize=markersize, label='Hmin-C') # distance between the electron center and its closest H
        plt.plot(5 * dOHmax - 7, '.', markersize=markersize, label='5 * dOHmax - 7', alpha=0.5)
        plt.legend()
    reacted = dOHyd > 1.5
    idx_react = jnp.argsort(-1 * reacted)[0]
    return valid, idx_react, Hyd_idx, Hyd, O, H, dHydC, dOC, dHCmin, dOHyd, dOHyd_2nd, O_Hyd_idx, O_Hyd, dOHydHmin, dOHmax

jit_analyze_traj = jit(analyze_traj, static_argnames=('plot', 'check_bond'))

box = 12.4171
import shutil
def create_input(path, coord, idx):
    N = coord.shape[0] // 3
    shutil.copytree(path + 'base', path + str(idx), dirs_exist_ok=True)
    atom_type = ['O', 'H', 'H'] * N + ['H']
    with open(path + str(idx) + "/water.xyz", "w") as f:
        f.write(f"{len(coord)}\n\n")
        for i in range(len(coord)):
            f.write(f"{atom_type[i]} {coord[i][0]} {coord[i][1]} {coord[i][2]}\n")
def create_input_set(path, coords, start_idx):
    if coords.shape[1] // 3 == 64:
        ref_path = "/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_rerun/base/"
    elif coords.shape[1] // 3 == 128:
        ref_path = "/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_rerun/base128/"
    shutil.copytree(ref_path, path + "base", dirs_exist_ok=True)
    for idx, coord in enumerate(coords):
        create_input(path, coord, start_idx + idx)
    print(f"Finished creating inputs for range {start_idx} to {start_idx + len(coords) - 1}")

import os
def collect_data(prefix, l, r, spread_limit, low_limit=0.1, N=64):
    # for collecting DFT data from CP2K calculations
    # prefix = "rerun_5"
    path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_{prefix}"
    force, energy, coord, spread, wc, eig, center = [], [], [], [], [], [], []
    for i in range(l, r):
    # for i in range(1):
        try:
            with open(f"{path}/{i}/polaron-force-1_0.xyz") as f:
                # frc = f.readlines()[-(3*N+2):-1]
                frc = f.readlines()[-(3*N+1):-1]
                frc = np.array([x.split()[3:] for x in frc], dtype=float) * 8.2387235e-8 / 1.602176634e-9
            with open(f"{path}/{i}/output") as f:
                lines = f.readlines()
                energy_line = [idx for idx in range(len(lines)) if "Total energy: " in lines[idx]][-1]
                energy_line = float(lines[energy_line].split()[-1]) * 27.211386245988
            with open(f"{path}/{i}/polaron-HOMO_centers_s1-1_0.data") as f:
                lines = f.readlines()[-(4*N+1):]
                wc_lines = np.array([x.split()[1:] for x in lines], dtype=float)
            with open(f"{path}/{i}/polaron-HOMO_centers_s2-1_0.data") as f:
                lines = f.readlines()[-4*N:]
                wc_lines_2 = np.array([x.split()[1:] for x in lines], dtype=float)
            wc_lines = np.concatenate([wc_lines, wc_lines_2])
            with open(f'{path}/{i}/polaron-eig-1_0.MOLog') as f:
                eig.append(float(f.readlines()[4*N+4].split()[3]))
            force.append(frc)
            energy.append(energy_line)
            coord.append(np.genfromtxt(f"{path}/{i}/water.xyz", skip_header=2)[:, 1:])
            idx = wc_lines[:, -1].argmax()
            box = 12.4171 if N==64 else 15.6446
            center.append(wc_lines[idx, :3] % box)
            spread.append(wc_lines[:, -1].max())
            wc.append(wc_lines[:, :3])
        except:
            print(i)
    spread = np.array(spread)
    idx = np.arange(len(spread))[(spread < spread_limit) * (spread > low_limit)]
    # idx = np.arange(len(spread))[(spread < 25) * (all_selected_jac < 0.05)]
    print(f"Spread < {spread_limit}: {len(idx)} out of {len(spread)}")
    spread = spread[idx]
    energy = np.array(energy)[idx]
    force = np.array(force).reshape(len(coord), -1)[idx]
    coord = np.array(coord).reshape(len(coord), -1)[idx]
    wc = np.array(wc)[idx]
    eig = np.array(eig)[idx]
    center = np.array(center)[idx]
    box = box * np.eye(3).reshape(9) * np.ones((coord.shape[0], 1))
    target_path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_reaction_data/{prefix}/set.001"
    os.makedirs(target_path, exist_ok=True)
    print(energy.shape, force.shape, coord.shape, spread.shape, box.shape, wc.shape, eig.shape, center.shape)
    # type_idx_path = "/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_reaction_data/rerun/type.raw"
    # shutil.copyfile(type_idx_path, f"{target_path}/../type.raw")
    np.savetxt(f"{target_path}/../type.raw", np.array([0,1,1] * N + [1]), fmt='%d')
    np.save(f"{target_path}/energy.npy", energy)
    np.save(f"{target_path}/force.npy", force)
    np.save(f"{target_path}/coord.npy", coord)
    np.save(f"{target_path}/spread.npy", spread)
    np.save(f"{target_path}/box.npy", box)
    np.save(f"{target_path}/wc.npy", wc)
    np.save(f"{target_path}/eig.npy", eig)
    np.save(f"{target_path}/atomic_center.npy", center)
    process_dipole(target_path)

def process_dipole(path):
    box = 12.4171
    O = np.load(f'{path}/coord.npy').reshape(-1, 193, 3)[:, :-1:3]
    wc = np.load(f'{path}/wc.npy').reshape(-1, 513, 3)
    center = np.load(f'{path}/atomic_center.npy').reshape(-1, 1, 3)
    dOW = jnp.linalg.norm((jnp.array(O)[:,:,None] - jnp.array(wc)[:,None,:] - box/2) % box - box/2, axis=-1)
    idx = jnp.argsort(dOW, axis=-1)[:,:,:8].reshape(-1, 512)
    dOW_sorted = jnp.sort(dOW, axis=-1)[:,:,:8].reshape(-1, 512)
    print('Max (O-W distance (1st - 2nd W)):', (dOW_sorted[:,1::2] - dOW_sorted[:,::2]).max())
    print('Max O-W distance:', dOW_sorted.max())
    wc_sorted = jnp.take_along_axis(wc, idx[:, :, None], axis=1)
    print('Min WC-e:', jnp.linalg.norm(wc_sorted - center, axis=-1).min())
    dipole = ((wc_sorted.reshape(-1, 64, 8, 3) - O[:,:,None] - box/2) % box - box/2).mean(-2)
    np.save(f'{path}/atomic_dipole.npy', dipole.reshape(-1, 64 * 3))

import jax
# @jax.jit
def analyze_traj_H(coords):
    N = coords.shape[1] // 3
    box = [12.4171, 15.6446, 19.711, 24.8342][int(np.log2(N))-6]
    O = jnp.array(coords[:, ::3][:, :-1])
    H = jnp.concatenate([coords[:,1::3], coords[:,2::3], coords[:,-1:]], axis=1)
    distOH = jnp.linalg.norm((O[:,:,None] - H[:,None] + box/2) % box - box/2, axis=-1)
    H2_idx = jnp.argsort(distOH, axis=-1)[:,:,:2].reshape(-1,2*N)
    # Hyd is the one H left after removing the two H closest to O
    H_is_bonded = vmap(jnp.isin, in_axes=(None, 0))(jnp.arange(2*N+1), H2_idx)
    # valid = True
    try:
        assert (H_is_bonded.sum(-1) == 2*N).all()
    except AssertionError:
        breakage_idx = jnp.where(H_is_bonded.sum(-1) < 2*N)[0]
        print('Warning: Bond breakage detected at ', breakage_idx)
        valid = False
    valid = (H_is_bonded.sum(-1) == 2*N).all()
    dOH_bonded = jnp.sort(distOH, axis=-1)[:,:,:2]
    # print('OH-bond:', dOH_bonded.min(), dOH_bonded.max())
    Hyd_idx = H_is_bonded.argsort(axis=-1)[:,0]
    Hyd = jnp.take_along_axis(H, Hyd_idx[:,None,None], axis=1)[:,0]
    dHydO = jnp.linalg.norm((O - Hyd[:,None] + box/2) % box - box/2, axis=-1)
    # O_Hyd is the O closest to Hyd
    O_Hyd_idx = dHydO.argmin(-1)
    dOHyd = dHydO.min(-1)
    dOHyd_2nd = jnp.sort(dHydO, axis=-1)[:,1]
    O_Hyd = jnp.take_along_axis(O, O_Hyd_idx[:,None,None], axis=1)[:,0]
    return valid, Hyd_idx, Hyd, O, H, dOHyd, dOHyd_2nd, O_Hyd_idx, O_Hyd


def get_Hyd(coords):
    O = coords[:-1:3]
    N = O.shape[0]
    H = jnp.concatenate([coords[1::3], coords[2::3], coords[-1:]], axis=0)
    distOH = jnp.linalg.norm((O[:,None] - H[None] + box/2) % box - box/2, axis=-1)
    H2_idx = jnp.argsort(distOH, axis=-1)[:,:2].reshape(2*N)
    H_is_bonded = jnp.isin(jnp.arange(2*N+1), H2_idx)
    return H[jnp.argsort(H_is_bonded)[0]]

def get_VH(coords, wc):
    N = coords.shape[0] // 3
    q = jnp.array([6,1,1]*N + [1] + [-8]*N)
    box = [12.4171, 15.6446, 19.711, 24.8342][int(np.log2(N))-6]
    M = [64, 64, 64] if box < 16 else [96, 96, 96]
    box3 = box * jnp.ones(3)
    cube_idx = (jnp.stack(jnp.meshgrid(*([jnp.array([-1,0,1])]*3),indexing='ij'))).reshape(3,27)
    MM = jnp.array(M).reshape(3,1,1,1)
    kgrid = jnp.stack(jnp.meshgrid(*[jnp.arange(m) for m in M], indexing='ij'))
    kgrid = 2*jnp.pi/box3[:,None,None,None] * ((kgrid-MM/2)%MM-MM/2)
    ksquare = (kgrid ** 2).sum(0)
    z = kgrid * (box3/jnp.array(M))[:,None,None,None] / 2
    sinz = jnp.sin(z)
    w3k = jnp.prod(jnp.where(z==0, 1, (sinz/z)**3), axis=0)
    Sk = jnp.prod(1 - sinz**2 + 2/15*sinz**4, axis=0)
    kfactor = -(14.399645*4*jnp.pi*jnp.prod(jnp.array(M))/jnp.prod(box3)) * jnp.exp(-ksquare/(4*2.0**2)) * w3k/(Sk*ksquare)
    kfactor = kfactor.at[0,0,0].set(0.)
    coord_N3 = jnp.concatenate([coords, wc])
    grid = jnp.zeros(M)
    M3 = jnp.array(M)
    coord_3N = ((coord_N3 % box3) / box3 * M3).T
    center_idx_3N = jnp.rint(coord_3N).astype(int) # in [0, M3]
    r_3N = coord_3N - center_idx_3N # lies in (-0.5, 0.5)
    fr_33N = jnp.stack([(r_3N-0.5)**2/2,
                        0.75 - r_3N**2,
                        (r_3N+0.5)**2/2]) # TSC assignment
    fr_27N = (fr_33N[:,None,None,0,:]*fr_33N[:,None,1,:]*fr_33N[:,2,:]).reshape(27,-1)
    all_idx = (center_idx_3N[:,None] + cube_idx[:,:,None]).reshape(3,-1) % M3[:,None]
    grid = grid.at[tuple(all_idx)].add((q*fr_27N).reshape(-1))
    V = jnp.fft.ifftn(jnp.fft.fftn(grid) * kfactor).real
    Hyd = get_Hyd(coords)
    grid_H = (Hyd / box * jnp.array(M)).astype(int)
    VH = jnp.fft.ifftn(jnp.fft.fftn(V) * jnp.exp(1j * (Hyd[:,None,None,None] * kgrid).sum(0)))[0,0,0].real
    return VH