from deepmd_jax.train import train, evaluate
from deepmd_jax.md import Simulation
from scripts.utils import jit_analyze_traj
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--model_path", type=str, default='polaron_reaction_newrefnew_1_uncompressed.pkl')
parser.add_argument("--temp", type=float, default=350)
parser.add_argument("--grid_size", type=int, default=None)
parser.add_argument("--max_block", type=int, default=1500)
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--expand_factor", type=float, default=1.0)
parser.add_argument("--run_idx", type=int, default=0)
parser.add_argument("--diffusion", action='store_true', default=False)
base_path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_reaction_data'
args = parser.parse_args()
if args.grid_size is None:  # N = 64, 128, 256, 512
    args.grid_size = [48, 64, 80, 96][int(np.log2(args.N))-6]
print("Simulation with parameters:", args)
index = args.index
box = [12.4171, 15.6446, 19.711, 24.8342][int(np.log2(args.N))-6]
if args.N == 64:
    path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_reaction_data/rerun'
    idx = np.load("idx_electron.npy")
    idx = np.random.choice(idx, size=1, replace=False)[0]
    initial_position = np.load(f'{path}/set.001/coord.npy')[idx].reshape(-1, 3)
elif args.N == 128:
    path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/extend_128_4/data.npz'
    idx = np.random.choice(3500, size=1, replace=False)[0]
    data = dict(np.load(path))
    initial_position = data['coord'][idx].reshape(-1, 3)
elif args.N == 256 and args.run_idx == 0:
    path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/old_init/testconf-H_e_256_fix_H_loose_new_continued_new_release/polaron-pos-1.xyz'
    with open(path, 'r') as f:
        lines = f.readlines()[-(256*3+1):]
        initial_position = np.array([x.split()[1:] for x in lines]).astype(float)
elif args.N == 256 and args.run_idx > 0:
    path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_{index}_{args.run_idx - 1}.npz"
    initial_position = dict(np.load(path))['positions'][-1]
elif args.N == 512 and args.run_idx == 0:
    idx = np.random.choice(500, size=1, replace=False)[0]
    initial_position = dict(np.load("extend_512_traj_equilibrate.npz"))['position'][300 + idx]
elif args.N == 512 and args.run_idx > 0:
    path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_{index}_{args.run_idx - 1}.npz"
    initial_position = dict(np.load(path))['positions'][-1]
box = box / args.expand_factor**(1/3)
initial_position = initial_position / args.expand_factor**(1/3)
sim = Simulation(
    model_path = 'trained_models/' + args.model_path,  # Path to the trained model
    grid_size = args.grid_size,
    box=box,                           # Angstroms
    type_idx=np.array([0,1,1]*args.N + [1]),
    mass=[15.9994, 1.0078],            # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,                            # femtoseconds
    tau_t=100,
    initial_position=initial_position, # Angstroms
    temperature=args.temp,               # Kelvin
    model_deviation_paths=['trained_models/' + args.model_path[:-5] + '1.pkl',
                            'trained_models/' + args.model_path[:-5] + '2.pkl'],
                            # 'trained_models/' + args.model_path[:-5] + '3.pkl',],
                            # 'trained_models/' + args.model_path[:-5] + '4.pkl'],
    report_interval=100 if not args.diffusion else 10,
    seed=np.random.randint(0, 1000000),
)

checkpoint = [sim.getPosition()]
checkpoint_center = []
invalid_positions = []
valid_flag = True
invalid_count = 0
import time
tic = time.time()
for i in range(args.max_block):
    trajectory = sim.run(2000)
    if not args.diffusion:
        valid, idx_react, Hyd_idx, Hyd, O, H, dHydC, dOC, dHCmin, dOHyd, dOHyd_2nd, O_Hyd_idx, O_Hyd, dOHydHmin, dHO12 = jit_analyze_traj(
            trajectory['position'][-1:], trajectory['center'][-1:], box=box, plot=False, check_bond=False)
        valid = (valid & (dHCmin > 0.3) & (dOHyd < 1.5) & (np.array([float(x[-2]) for x in sim.log[-len(trajectory['center'])//3:]]).mean() < 0.5)).item()
    else:
        valid, idx_react, Hyd_idx, Hyd, O, H, dHydC, dOC, dHCmin, dOHyd, dOHyd_2nd, O_Hyd_idx, O_Hyd, dOHydHmin, dHO12 = jit_analyze_traj(
            trajectory['position'][::10], trajectory['center'], box=box, plot=False, check_bond=False)
        valid = (valid & (dOC.min() > 2.5)).item()
    if valid:
        invalid_count = 0
        checkpoint.append(trajectory['position'][-1])
        checkpoint_center.append(trajectory['center'][-1])
    else:
        invalid_positions.append(len(checkpoint))
        if invalid_count < 3:
            sim.step -= 2000
            if len(checkpoint) > 1:
                invalid_count += 1
        else:
            checkpoint.pop()
            checkpoint_center.pop()
            sim.step -= 2000 * 2
            invalid_count = 0
        sim.setPosition(checkpoint[-1])
        sim.setRandomVelocity(args.temp)
    print(f'Iteration {i}, valid: {valid}, invalid_count: {invalid_count}, checkpoint size: {len(checkpoint)}')
    if args.N < 256 and time.time() - tic > 3600 * 23.5:
        print("Time limit reached, stop simulation.")
        break
    elif args.N == 256 and time.time() - tic > 3600 * 10:
        print("Time limit reached, stop simulation.")
        break
    elif args.N == 512 and time.time() - tic > 3600 * 5.5:
        print("Time limit reached, stop simulation.")
        break
if args.N < 256:
    np.savez(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_{index}.npz', positions=np.array(checkpoint)[1:], centers=np.array(checkpoint_center), invalid_positions=np.array(invalid_positions))
else:
    np.savez(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_{index}_{args.run_idx}.npz', positions=np.array(checkpoint)[1:], centers=np.array(checkpoint_center), invalid_positions=np.array(invalid_positions))