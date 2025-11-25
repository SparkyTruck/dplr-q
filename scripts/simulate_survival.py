from deepmd_jax.train import train, evaluate
from deepmd_jax.md import Simulation
from scripts.utils import jit_analyze_traj
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--model_path", type=str, default='polaron_reaction_newrefnew_1_uncompressed.pkl')
parser.add_argument("--temp", type=int, default=350)
parser.add_argument("--grid_size", type=int, default=None)
parser.add_argument("--max_blocks", type=int, default=50)
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--total_run", type=int, default=None)
parser.add_argument("--record_traj", action='store_true', default=False)
parser.add_argument("--expand_factor", type=float, default=1.0)
parser.add_argument("--D", action='store_true', default=False)
parser.add_argument("--qsd", action='store_true', default=True)
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
elif args.N == 256:
    path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/old_init/testconf-H_e_256_fix_H_loose_new_continued_new_release/polaron-pos-1.xyz'
    with open(path, 'r') as f:
        lines = f.readlines()[-(256*3+1):]
        initial_position = np.array([x.split()[1:] for x in lines]).astype(float)
elif args.N == 512:
    idx = np.random.choice(500, size=1, replace=False)[0]
    initial_position = dict(np.load("extend_512_traj_equilibrate.npz"))['position'][300 + idx]
box = box / args.expand_factor**(1/3)
initial_position = initial_position / args.expand_factor**(1/3)
sim = Simulation(
    model_path = 'trained_models/' + args.model_path,  # Path to the trained model
    grid_size = args.grid_size,
    box=box,                           # Angstroms
    type_idx=np.array([0,1,1]*args.N + [1]),
    mass=[15.9994, 1.0078 if not args.D else 2.014],  # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,                            # femtoseconds
    tau_t=100,
    initial_position=initial_position, # Angstroms
    temperature=args.temp,               # Kelvin
    model_deviation_paths=['trained_models/' + args.model_path[:-5] + '1.pkl',
                            'trained_models/' + args.model_path[:-5] + '2.pkl'],
                            # 'trained_models/' + args.model_path[:-5] + '3.pkl',],
                            # 'trained_models/' + args.model_path[:-5] + '4.pkl'],
    report_interval=100 if not args.record_traj else 10,
    seed=np.random.randint(0, 1000000),
)

all_timing = []
all_traj_record = []
all_center_record = []
block = 2000
max_blocks = args.max_blocks
if not args.qsd:
    init_coords = np.load(f"init_coords_{args.N}_{args.temp}.npy")
else:
    init_coords = np.load(f"init_coords_qsd_{args.N}_{args.temp}.npy")
if args.expand_factor < 1.0:
    init_coords = np.load(f"init_coords_expand_{args.N}_{args.temp}.npy")
# np.random.seed(42)
np.random.shuffle(init_coords)
total_run = 100 if args.total_run is None else args.total_run
import time
tic = time.time()
for i in range(total_run):
# for i in range(total_run*index, total_run*(index+1)):
    print(f"############################## Starting simulation {i} ##############################")
    sim.setPosition(init_coords[i])
    sim.setRandomVelocity(args.temp)
    sim.step = 0
    reacted = False
    valid = True
    traj_record = []
    center_record = []
    for ii in range(max_blocks):
        trajectory = sim.run(block)
        valid, idx_react, Hyd_idx, Hyd, O, H, dHydC, dOC, dHCmin, dOHyd, dOHyd_2nd, O_Hyd_idx, O_Hyd, dOHydHmin, dHO12 = jit_analyze_traj(
            trajectory['position'][-1:], trajectory['center'][-1:], box=box, plot=False, check_bond=False)
        valid = valid & (np.array([float(x[-2]) for x in sim.log[-len(trajectory['center'])//3:]]).mean() < 0.5)
        if not valid:
            print("Unphysical structure encountered. Stopping simulation.")
            all_timing.append(-1)
            break
        reacted = ((dHCmin < 0.3) & (dOHyd > 1.5)).item()
        if reacted:
            idx_react = jit_analyze_traj(trajectory['position'], centers=None, box=box, plot=False, check_bond=False)[1]
            timing = idx_react.item() + block * ii
            all_timing.append(timing)
            print(f"Reaction happened at step {timing}")
            if args.record_traj:
                traj_record.append(trajectory['position'][:idx_react.item()+1:10])
                center_record.append(trajectory['center'][:(idx_react.item()+1)//10])
            break
        if args.record_traj:
            traj_record.append(trajectory['position'][::10])
            center_record.append(trajectory['center'])
        if ii == max_blocks - 1:
            all_timing.append(-2)
            print(f"Reaction did not happen within {max_blocks * block} steps.")
    if valid and args.record_traj:
        all_traj_record.append(np.concatenate(traj_record, axis=0))
        all_center_record.append(np.concatenate(center_record, axis=0))
        # all_traj_record.append(np.concatenate(traj_record, axis=0)[-4000:])
    if args.N == 64:
        max_time = 72000  # 20 hours
    elif args.N == 128:
        max_time = 3600 * 22  # 22 hours
    elif args.N == 256:
        max_time = 3600 * 12  # 12 hours
    else:  # N == 512
        max_time = 3600 * ({'550': 5.5, '500': 5, '450':4.5, '400':4, '350':3.5}[str(args.temp)])
    if time.time() - tic > max_time:
    # if time.time() - tic > (72000 if args.N < 128 else 3600 * ({'550': 5.5, '500': 5, '450':4.5, '400':4, '350':3.5}[str(args.temp)])):
        print("Approaching time limit, stopping further simulations.")
        break
print(all_timing)
print("Finished all simulations; total trajectories:", len(all_timing))
if args.qsd:
    if args.record_traj:
        np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_time_qsd_traj_{args.N}_{args.temp}_{index}.npy', np.array(all_timing))
        for ii in range(len(all_traj_record)):
            record = all_traj_record[ii]
            center_record = all_center_record[ii]
            np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_qsd_traj_{args.N}_{args.temp}_{index}_{ii}.npy', record)
            np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_qsd_center_{args.N}_{args.temp}_{index}_{ii}.npy', center_record)
    else:
        np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_time_qsd_{args.N}_{args.temp}_{index}.npy', np.array(all_timing))
if args.record_traj:
    # all_traj_record = np.array(all_traj_record)
    if not args.D:
        for ii, record in enumerate(all_traj_record):
            np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_traj_{args.N}_{args.temp}_{index}_{ii}.npy', record)
        np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_time_traj_{args.N}_{args.temp}_{index}.npy', np.array(all_timing))
    else:
        for ii, record in enumerate(all_traj_record):
            np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_traj_D_{args.N}_{args.temp}_{index}_{ii}.npy', record)
        np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_time_traj_D_{args.N}_{args.temp}_{index}.npy', np.array(all_timing))
else:
    np.save(f'/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_sim/traj/survival_time_{args.N}_{args.temp}_{index}.npy', np.array(all_timing))