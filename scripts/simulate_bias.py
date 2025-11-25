from deepmd_jax.train import train, evaluate
from deepmd_jax.md import Simulation
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--step", type=int, default=100000)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--temp", type=float, default=350)
parser.add_argument("--grid_size", type=int, default=None)
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--run_idx", type=int, default=0)
parser.add_argument("--save_traj", action='store_true', default=False)
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
    path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_bias_{index}_{args.run_idx - 1}.npz"
    initial_position = dict(np.load(path))['last_position']
elif args.N == 512 and args.run_idx == 0:
    idx = np.random.choice(500, size=1, replace=False)[0]
    initial_position = dict(np.load("extend_512_traj_equilibrate.npz"))['position'][300 + idx]
elif args.N == 512 and args.run_idx > 0:
    path = f"/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_bias_{index}_{args.run_idx - 1}.npz"
    initial_position = dict(np.load(path))['last_position']
sim = Simulation(
    model_path = 'trained_models/' + args.model_path,  # Path to the trained model
    grid_size = args.grid_size,
    box=box,                           # Angstroms
    bias=True,
    type_idx=np.array([0,1,1]*args.N + [1]),
    mass=[15.9994, 1.0078],            # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,                            # femtoseconds
    tau_t=100,
    initial_position=initial_position, # Angstroms
    temperature=args.temp,               # Kelvin
    report_interval=10,
    seed=np.random.randint(0, 1000000),
    model_deviation_paths=['trained_models/' + args.model_path[:-5] + '1.pkl',
                            'trained_models/' + args.model_path[:-5] + '2.pkl',],
)
# position_traj = []
center_traj = []
cv_traj = []
traj = []
model_devi_traj = []
block = 2000
from scripts.utils import jit_analyze_traj
total_valid = 0
checkpoint = sim.getPosition()
import time
tic = time.time()
for i in range(args.step // block):
    print(f"####### Running block {i}, current valid count {total_valid} #######")
    trajectory = sim.run(block)
    valid = jit_analyze_traj(trajectory['position'][-1:], trajectory['center'][-1:], plot=False, check_bond=False)[0]
    valid *= np.array([x[-2] for x in sim.log[-block//10:]]).max() < 1
    total_valid += valid
    print(f"####### Block {i} done, valid: {valid} #######")
    if valid:
        # position_traj.append(trajectory['position'][::10])
        center_traj.append(trajectory['center'])
        cv_traj.append(trajectory['cv'])
        if args.save_traj:
            traj.append(trajectory['position'][::10])
        model_devi_traj.append(np.array([x[-2] for x in sim.log[-len(trajectory['cv']):]]))
        checkpoint = sim.getPosition()
    else:
        print("####### Invalid configuration detected, rolling back to last checkpoint #######")
        sim.setPosition(checkpoint)
        sim.setRandomVelocity(args.temp)
        sim.step -= block
    if time.time() - tic > 3600 * (23 if args.N < 256 else (11.5 if args.N == 256 else 5.5)):
        print("####### Time limit approaching, stopping simulation #######")
        break
# position_traj = np.concatenate(position_traj, axis=0)
center_traj = np.concatenate(center_traj, axis=0)
cv_traj = np.concatenate(cv_traj, axis=0)
model_devi_traj = np.concatenate(model_devi_traj, axis=0)
if args.save_traj:
    traj = np.concatenate(traj)
if args.N < 256:
    np.savez(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_bias_{index}.npz', center=center_traj, cv=cv_traj, model_devi=model_devi_traj, traj=traj)
else:
    np.savez(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/sim_bias_{index}_{args.run_idx}.npz', last_position=sim.getPosition(), center=center_traj, cv=cv_traj)