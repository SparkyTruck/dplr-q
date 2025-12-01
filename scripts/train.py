from deepmd_jax.train import train
import argparse
# import jax
# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=1)
parser.add_argument("--data", type=str, default='rerun')
parser.add_argument("--wannier", action='store_true', default=False)
parser.add_argument("--water", action='store_true', default=False)
parser.add_argument("--old_electron", action='store_true', default=False)
parser.add_argument("--Hplus", action='store_true', default=False)
parser.add_argument("--s_pref_e", type=float, default=0.01)
parser.add_argument("--s_pref_f", type=float, default=1.0)
parser.add_argument("--l_pref_e", type=float, default=1.0)
parser.add_argument("--l_pref_f", type=float, default=0.01)
parser.add_argument("--centroid", action='store_true', default=False)
parser.add_argument("--model_name", type=str, default='polaron_reaction_model')
parser.add_argument("--w_path", type=str, default='wannier_centroid_4.pkl')
parser.add_argument("--step", type=int, default=1000000)
base_path = '/pscratch/sd/r/ruiqig/polaron_cp2k/test-conf-reaction/polaron_reaction_data'
args = parser.parse_args()
idx = args.idx
if args.data == 'full':
    train_data_path = f'{base_path}/train-full'
    val_data_path = f'{base_path}/val-full'
elif args.data == 'valtest':
    train_data_path = f'{base_path}/validation/train'
    val_data_path = f'{base_path}/validation/val'
elif args.data == 'valtest_full':
    train_data_path = [f'{base_path}/validation/train', f'{base_path}/validation/val']
    val_data_path = None
elif args.data == 'electron':
    train_data_path = f'{base_path}/electron'
    val_data_path = f'{base_path}/electron' if args.wannier else None
elif args.data == 'radical':
    train_data_path = f'{base_path}/radical'
    val_data_path = f'{base_path}/radical' if args.wannier else None
elif args.data == 'rerun':
    train_data_path = f'{base_path}/rerun'
    val_data_path = f'{base_path}/rerun' if args.wannier else f'{base_path}/rerun/spread/'
elif 'renew' in args.data:
    subsets = args.data[5:]
    train_data_path = [f'{base_path}/rerun/filter_20'] + [f'{base_path}/active_{subset}' for subset in subsets]
    val_data_path = train_data_path if args.wannier else None
elif 'rerun' in args.data:
    subsets = args.data[5:]
    train_data_path = [f'{base_path}/rerun'] + [f'{base_path}/rerun_{subset}' for subset in subsets]
    val_data_path = train_data_path if args.wannier else f'{base_path}/rerun/spread/'
elif 'final' in args.data:
    subsets = args.data[5:]
    train_data_path = [f'{base_path}/rerun/solve_2_2_0.6/filter_20'] + [f'{base_path}/rerun/solve_2_2_0.6/active_{subset}' for subset in subsets]
    val_data_path = train_data_path if args.wannier else None
elif 'w0123' in args.data:
    subsets = args.data[5:]
    train_data_path = [f'{base_path}/rerun/w0123/filter_20'] + [f'{base_path}/rerun/w0123/active_{subset}' for subset in subsets]
    val_data_path = train_data_path if args.wannier else None
elif 'dense' in args.data:
    subsets = args.data[5:]
    train_data_path = [f'{base_path}/rerun/solve_2_2_0.6/dense/filter_20'] + [f'{base_path}/rerun/solve_2_2_0.6/dense/active_{subset}' for subset in subsets]
    val_data_path = train_data_path if args.wannier else None
elif args.data == 'train':
    train_data_path = f'{base_path}/rerun/train'
    val_data_path = f'{base_path}/rerun/val'
elif args.data == 'water':
    train_data_path = '/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128/'
    val_data_path = '/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128_val/'
elif args.data == '0.3':
    train_data_path = f'{base_path}/rerun/short_0.3'
    val_data_path = None
elif args.data == '0.4':
    train_data_path = f'{base_path}/rerun/short_0.4'
    val_data_path = None
elif 'solve' in args.data:
    train_data_path = f'{base_path}/rerun/{args.data}'
    val_data_path = None
elif args.data == 'old':
    train_data_path = f'{base_path}/old_train'
    val_data_path = f'{base_path}/old_train' if args.wannier else None
elif args.data == 'W':
    train_data_path = f'{base_path}/rerun/W'
    val_data_path = f'{base_path}/rerun/W' if args.wannier else None
elif 'W' in args.data:
    subsets = args.data[1:]
    train_data_path = [f'{base_path}/rerun/W'] + [f'{base_path}/rerun_{subset}/W' for subset in subsets]
    val_data_path = train_data_path if args.wannier else None
elif args.data == 'Hplus':
    train_data_path = f'{base_path}/../Hplus_data_full'
    val_data_path = f'{base_path}/../Hplus_data_full' if args.wannier else None
if args.water:
    train(
        model_type='energy',                   # Model type
        rcut=6,                              # Cutoff radius
        # mp=True,
        save_path=f'trained_models/water_{idx}.pkl',                 # Path to save the trained model
        train_data_path=[base_path + "/water_128", base_path + "/water_128_val"],
        step=args.step,
    )
elif args.old_electron:
    train(
        model_type='dplr_q1',                   # Model type
        rcut=6,                              # Cutoff radius
        mp=True,
        save_path=f'trained_models/old_electron_{idx}.pkl',                 # Path to save the trained model
        train_data_path=base_path + "/polaron_full_train",
        val_data_path=base_path + "/polaron_full_val",
        val_batch_size_ratio=128,
        step=args.step,
        dplr_wannier_model_path = f'trained_models/{args.w_path}',
        dplr_q_atoms = [6, 1],
        dplr_q_wc = [-8],
        dplr_beta = 0.6,
        dplr_q1_beta = 2.0,
    )
elif args.Hplus:
    train(
        model_type='energy',                   # Model type
        # mp=True,
        rcut=6,                              # Cutoff radius
        save_path=f'trained_models/Hplus_0_{idx}.pkl',                 # Path to save the trained model
        train_data_path=[f'{base_path}/../Hplus_data_full',
                            f'{base_path}/Hplus_active_0'],
        step=args.step,
    )
elif not args.wannier and not args.centroid:
    train(
        model_type='dplr_q1',                   # Model type
        rcut=6,                              # Cutoff radius
        mp=True,
        # embed_widths=[48,48,96],
        # embed_mp_widths=[96,96,96],
        # axis_neurons=16,
        # fit_widths=[192,192,192],
        save_path=f'trained_models/{args.model_name}_{idx}.pkl',                 # Path to save the trained model
        train_data_path=train_data_path,
        # train_data_path=[f'{base_path}/rerun', f'{base_path}/rerun_5'],
        val_data_path=val_data_path,
        val_batch_size_ratio=128,
        step=args.step,
        dplr_wannier_model_path = f'trained_models/{args.w_path}',
        dplr_q_atoms = [6, 1],
        dplr_q_wc = [-8],
        dplr_beta = 0.6,
        dplr_q1_beta = 2.0,
        # batch_size=4,
        s_pref_e=args.s_pref_e,
        s_pref_f=args.s_pref_f,
        l_pref_e=args.l_pref_e,
        l_pref_f=args.l_pref_f,
    )
elif args.wannier:
    train(
      model_type='atomic_iter',     
      rcut=6.0,        
      mp=True,
      save_path=f'trained_models/{args.model_name}_{idx}.pkl',
      train_data_path=train_data_path,
      val_data_path=val_data_path,
      step=args.step,
      gamma_iter=2,
      n_iter=4,
      perturb_iter=1,
      batch_size=32,
      val_batch_size_ratio=128,
      lr=0.01,
      print_every=1000,
      compress=False,
    )
elif args.centroid:
    train(
      model_type='atomic',     
      rcut=6.0,        
      atomic_sel=[0],
      mp=True,
      step=args.step,
      save_path=f'trained_models/{args.model_name}_{idx}.pkl',
    #   save_path=f'trained_models/wannier_centroid_{idx}.pkl',
      train_data_path=train_data_path,
      val_data_path=val_data_path,
    )