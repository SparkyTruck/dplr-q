import jax
import optax
import numpy as np
import jax.numpy as jnp
import time, datetime
import flax.linen as nn
from functools import partial
from .utils import get_p3mlr_fn, get_p3mlr_grid_size, load_model, save_model, compress_model, save_dataset, get_schrodinger_1e_fn
from .data import DPDataset
from .dpmodel import DPModel
# from .dpmodel_iterw import DPModel_iterw as DPModel
from typing import Union, List
import tempfile
import os

def train(
    model_type: str,
    rcut: float,
    train_data_path: Union[str, List[str]],
    val_data_path: Union[str, List[str]] = None,
    save_path: str = 'model.pkl',
    step: int = 1000000,
    mp: bool = False,
    atomic_sel: List[int] = None,
    embed_widths: List[int] = [32,32,64],
    embed_mp_widths: List[int] = [64,64,64],
    fit_widths: List[int] = None,
    axis_neurons: int=12,
    lr: float = None,
    batch_size: int = None,
    val_batch_size_ratio: int = 4,
    compress: bool = True,
    print_every: int = 1000,
    atomic_data_prefix: str = 'atomic_dipole',
    s_pref_e: float = 0.02,
    l_pref_e: float = 1,
    s_pref_f: float = 1000,
    l_pref_f: float = 1,
    dplr_wannier_model_path: str = None,
    dplr_q_atoms: List[float] = None,
    dplr_q_wc: List[float] = None,
    dplr_beta: float = 0.4,
    dplr_q1_beta: float = 2,
    dplr_resolution: float = 5,
    lr_limit: float = 1e-6,
    beta2: float = 0.99,
    decay_steps: int = 5000,
    getstat_bs: int = 64,
    label_bs: int = 256,
    tensor_2nd: bool = True,
    print_loss_smoothing: int = 20,
    compress_Ngrids: int = 1024,
    compress_r_min: float = 0.6,
    seed: int = None,
    gamma_iter: float = 2,
    n_iter: int = 4,
    perturb_iter: float = 1.,
):
    '''
        Entry point for training deepmd-jax models.
        Input arguments:
            model_type: 'energy' (force field), 'atomic' (e,g. Deep Wannier), 'dplr' (long-range), 'atomic_iter' (Wannier center iteration), or 'dplr_q1' (dplr-q with 1e- schrodinger).
            rcut: cutoff radius (Angstrom) for the model.
            save_path: path to save the trained model.
            train_data: path to training data (str) or list of paths to training data (List[str]).
            val_data: path to validation data (str) or list of paths to validation data (List[str]).
            step: number of training steps. Depending on dataset size, expect 1e5-1e7 for energy models and 1e5-1e6 for wannier models.
            mp: whether to use message passing model for more accuracy at a higher cost.
            atomic_sel: Selects the atom types for prediction. Only used when model_type == 'atomic'.
            embed_widths: Widths of the embedding neural network.
            embed_mp_widths: Widths of the embedding neural network in message passing. Only used when mp == True.
            fit_widths: Widths of the fitting neural network.
            axis_neurons: Number of axis neurons to project the atomic features before the fitting network. Recommended range: 8-16.
            lr: learning rate at start. If None, default values (0.002 for 'energy' and 0.01 for 'atomic') is used.
            batch_size: training batch size in number of frames. If None, will be automatically determined by label_bs.
            val_batch_size_ratio: validation batch size / batch_size. Increase for stabler validation loss.
            compress: whether to compress the model after training for faster inference.
            print_every: interval for printing loss and validation.
            atomic_data_prefix: prefix for .npy label files when model_type == 'atomic'.
            s_pref_e: starting prefactor for energy loss.
            l_pref_e: limit prefactor for energy loss.
            s_pref_f: starting prefactor for force loss.
            l_pref_f: limit prefactor for force loss.
            dplr_wannier_model_path: path to the Deep Wannier model, only used in 'dplr'.
            dplr_q_atoms: charge of atomic cores for each atom type, only used in 'dplr'.
            dplr_q_wc: charge of wannier center/centroid for each type in atomic_sel of the wannier model, only used in 'dplr'.
            dplr_beta: inverse spread of the smoothed point charge distribution, only used in 'dplr'.
            dplr_resolution: higher resolution means denser grid: resolution = 1 / (grid length * beta); only used in 'dplr'.
            lr_limit: learning rate at end of training.
            beta2: adam optimizer parameter.
            decay_steps: learning rate exponentially decays every decay_steps.
            getstat_bs: batch size for computing model statistics at initialization.
            label_bs: training batch size in number of atoms.
            tensor_2nd: whether to use 2nd order tensor descriptor for more accuracy.
            print_loss_smoothing: smoothing factor for loss printing.
            compress_Ngrids: Number of intervals used in compression.
            compress_r_min: A safe lower bound for interatomic distance in the compressed model.
    '''
    
    TIC = time.time()
    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for training.')

    # width check
    if fit_widths is None:
        if 'atomic' not in model_type:
            fit_widths = [128, 128, 128]
        else:
            width = embed_mp_widths[-1] if mp else embed_widths[-1]
            fit_widths = [width, width, width]
    for i in range(len(embed_widths)-1):
        if embed_widths[i+1] % embed_widths[i] != 0:
            raise ValueError('embed_widths[i] must divide embed_widths[i+1]')
    if mp:
        if embed_widths[-1] != embed_mp_widths[0]:
            raise ValueError('embed_widths[-1] must equal embed_mp_widths[0].')
        for i in range(len(embed_mp_widths)-1):
            if embed_mp_widths[i+1] % embed_mp_widths[i] != 0 and embed_mp_widths[i+1] % embed_widths[i] != 0:
                raise ValueError('embed_mp_widths[i] must divide or be divisible by embed_mp_widths[i+1]')
    for i in range(len(fit_widths)-1):
        if fit_widths[i+1] != fit_widths[i] != 0:
            print('# Warning: it is recommended to use the same width for all layers in the fitting network.')
    if 'atomic' in model_type:
        if mp:
            if embed_mp_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic mp models, embed_mp_widths[-1] must equal fit_widths[-1].')
        else:
            if embed_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic models, embed_widths[-1] must equal fit_widths[-1].')
    if model_type == 'atomic_iter':
        compress = False

    # load dataset
    if model_type == 'energy' or model_type == 'dplr' or model_type == 'dplr_q1':
        labels = ['coord', 'box', 'force', 'energy']
    elif model_type == 'atomic':
        labels = ['coord', 'box', atomic_data_prefix]
        assert type(atomic_sel) == list, ' Must provide atomic_sel properly for model_type "atomic"'
    elif model_type == 'atomic_iter':
        labels = ['coord', 'box', "atomic_center"]
        atomic_sel = []
    else:
        raise ValueError('model_type should be "energy", "atomic", "dplr", "dplr_q1", or "atomic_iter".')
    if type(train_data_path) == str:
        train_data_path = [train_data_path]
    else:
        train_data_path = [[path] for path in train_data_path]
    train_data = DPDataset(train_data_path,
                           labels,
                           {'atomic_sel':atomic_sel})
    train_data.compute_lattice_candidate(rcut)
    use_val_data = val_data_path is not None
    if use_val_data:
        if type(val_data_path) == str:
            val_data_path = [val_data_path]
        else:
            val_data_path = [[path] for path in val_data_path]
        val_data = DPDataset(val_data_path,
                             labels,
                             {'atomic_sel':atomic_sel})
        val_data.compute_lattice_candidate(rcut)
    else:
        val_data = None

    # for dplr, convert dataset to short-range
    if 'dplr' in model_type:
        if type(dplr_wannier_model_path) is not str:
            raise ValueError('Must properly provide dplr_wannier_model_path (path to your trained Wannier model) for model_type involving "dplr".')
        if type(dplr_q_atoms) is not list:
            raise ValueError('Must properly provide dplr_q_atoms for model_type involving "dplr".')
        if type(dplr_q_wc) is not list:
            raise ValueError('Must properly provide dplr_q_wc for model_type involving "dplr".')
        wc_model, wc_variables = load_model(dplr_wannier_model_path, replicate=False)
        if len(dplr_q_wc) != len(wc_model.params['nsel']):
            raise ValueError('dplr_q_wc must correspond to atomic_sel of the Wannier model.')
        subsets = train_data.get_flattened_data()
        if use_val_data:
            subsets += val_data.get_flattened_data()
        print('# Building short-range dataset...', end='')
        tic_sr = time.time()
        for subset in subsets:
            process_long_range_subset(subset,
                                      model_type,
                                      dplr_q_atoms,
                                      dplr_q_wc,
                                      dplr_beta,
                                      dplr_q1_beta,
                                      dplr_resolution,
                                      wc_model,
                                      wc_variables)
        print(' Done. Time: %d s' % (time.time() - tic_sr))
            

    # construct model
    if model_type == 'energy' or 'dplr' in model_type:
        out_norm = 1
        nsel = None
        atomic_data_prefix = None
    elif model_type == 'atomic':
        out_norm = train_data.get_atomic_label_scale()
        nsel = atomic_sel
    elif model_type == 'atomic_iter':
        out_norm = 1/30.
        nsel = [train_data.ntypes]
        atomic_data_prefix = 'atomic_center'
    
    params = {
        'type': model_type,
        'atomic_data_prefix': atomic_data_prefix,
        'embed_widths': embed_widths[:-1] if mp else embed_widths,
        'embedMP_widths': embed_widths[-1:] + embed_mp_widths if mp else None,
        'fit_widths': fit_widths,
        'axis': axis_neurons,
        'Ebias': train_data.fit_energy() if 'atomic' not in model_type else None,
        'rcut': rcut,
        'use_2nd': tensor_2nd,
        'use_mp': mp,
        'atomic': 'atomic' in model_type,
        'atomic_iter': model_type == 'atomic_iter',
        'nsel': nsel,
        'out_norm': out_norm,
        **train_data.get_stats(rcut,
                               getstat_bs,
                               iterw=(model_type == 'atomic_iter')),
    }
    if 'dplr' in model_type:
        dplr_params = {
            'dplr_wannier_model_and_variables': (wc_model, wc_variables),
            'dplr_q_atoms': dplr_q_atoms,
            'dplr_q_wc': dplr_q_wc,
            'dplr_beta': dplr_beta,
            'dplr_resolution': dplr_resolution,
        }
        if model_type == 'dplr_q1':
            dplr_params['dplr_q1_beta'] = dplr_q1_beta
        params.update(dplr_params)
    model = DPModel(params)
    print('# Model params:', {k:v for k,v in model.params.items() if k != 'dplr_wannier_model_and_variables'})

    # initialize model variables
    batch, type_count, lattice_args = train_data.get_batch(1)
    if model_type == 'atomic_iter':
        def process_batch(batch):
            wc_init = np.random.randn(*batch["atomic"].shape)
            perturb = wc_init * np.random.uniform(size=(wc_init.shape[:2] + (1,))) * perturb_iter / np.linalg.norm(wc_init, axis=-1)[..., None]
            wc_init = batch['atomic'] + perturb
            batch['coord'] = np.concatenate([batch['coord'], wc_init], axis=1)
            static_args = nn.FrozenDict({'type_count': tuple(type_count) + (batch["atomic"].shape[1],), 'lattice': lattice_args})
            return batch, static_args
        batch, static_args = process_batch(batch)
    else:
        static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})
    if seed is None:
        seed = np.random.randint(65536)
    variables = model.init(
                    jax.random.PRNGKey(seed),
                    batch['coord'][0],
                    batch['box'][0],
                    static_args,
                )
    print('# Model initialized with parameter count %d.' %
           sum(i.size for i in jax.tree_util.tree_flatten(variables)[0]))
    
    # initialize optimizer
    if lr is None:
        lr = 0.002 if 'atomic' not in model_type else 0.01
    if step < decay_steps * 10:
        decay_steps = max(step // 10, 1)
    lr_scheduler = optax.exponential_decay(
                        init_value = lr,
                        transition_steps = decay_steps,
                        decay_rate = (lr_limit/lr) ** (decay_steps / (step-decay_steps)),
                        transition_begin = 0,
                        staircase = True,
                    )
    optimizer = optax.adam(learning_rate = lr_scheduler,
                           b2 = beta2)
    opt_state = optimizer.init(variables)
    print('# Optimizer initialized with initial lr = %.1e. Starting training %d steps...' % (lr, step))

    # define training step
    loss_fn, loss_and_grad_fn = model.get_loss_fn(gamma_iter=gamma_iter, n_iter=n_iter)
    # loss_fn_val, _ = model.get_loss_fn(gamma_iter=gamma_iter, n_iter=8, val=True)
    loss_fn_val, _ = model.get_loss_fn(gamma_iter=gamma_iter, n_iter=n_iter+3, val=True)
    if 'atomic' not in model_type:
        state = {'loss_avg': 0., 'le_avg': 0., 'lf_avg': 0., 'iteration': 0}
    elif model_type == 'atomic':
        state = {'loss_avg': 0., 'iteration': 0}
    elif model_type == 'atomic_iter':
        state = {'loss_avg': 0., 'iteration': 0, 'loss_avg_each_step': jnp.zeros(n_iter)}
    
    @partial(jax.jit, static_argnames=('static_args',))
    def train_step(batch, variables, opt_state, state, static_args):
        r = lr_scheduler(state['iteration']) / lr
        if 'atomic' not in model_type:
            pref = {'e': s_pref_e*r + l_pref_e*(1-r),
                    'f': s_pref_f*r + l_pref_f*(1-r)}
            (loss_total, (loss_e, loss_f)), grads = loss_and_grad_fn(variables,
                                                                    batch,
                                                                    pref,
                                                                    static_args)
            for key, value in zip(['loss_avg', 'le_avg', 'lf_avg'],
                                  [loss_total, loss_e, loss_f]):
                state[key] = state[key] * (1-1/print_loss_smoothing) + value
        elif model_type == 'atomic':
            loss_total, grads = loss_and_grad_fn(variables,
                                                 batch,
                                                 static_args)
            state['loss_avg'] = state['loss_avg'] * (1-1/print_loss_smoothing) + loss_total
        elif model_type == 'atomic_iter':
            (loss_total, loss_each_step), grads = loss_and_grad_fn(variables,
                                                            batch,
                                                            static_args)
            state['loss_avg'] = state['loss_avg'] * (1-1/print_loss_smoothing) + loss_total
            state['loss_avg_each_step'] = state['loss_avg_each_step'] * (1-1/print_loss_smoothing) + loss_each_step
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        state['iteration'] += 1
        return variables, opt_state, state
    
    # define validation step
    @partial(jax.jit, static_argnames=('static_args',))
    def val_step(batch, variables, static_args):
        if 'atomic' not in model_type:
            pref = {'e': 1, 'f': 1}
            _, (loss_e, loss_f) = loss_fn(variables,
                                          batch,
                                          pref,
                                          static_args)
            return loss_e, loss_f
        elif model_type == 'atomic':
            loss_total = loss_fn(variables,
                                 batch,
                                 static_args)
            return loss_total
        elif model_type == 'atomic_iter':
            loss_total, loss_each_step = loss_fn_val(variables,
                                                 batch,
                                                 static_args)
            return loss_each_step
        
    # configure batch size
    if batch_size is None:
        if model_type == 'atomic_iter':
            raise ValueError("must specify batch_size when model_type == 'atomic_iter'")
        print(f'# Auto batch size = int({label_bs}/nlabels_per_frame)')
    else:
        print(f'# Using batch size {batch_size}')
    def get_batch_train():
        if batch_size is None:
            return train_data.get_batch(label_bs, 'label')
        else:
            return train_data.get_batch(batch_size)
    def get_batch_val():
        ret = []
        for _ in range(val_batch_size_ratio):
            if batch_size is None:
                ret.append(val_data.get_batch(label_bs, 'label'))
            else:
                ret.append(val_data.get_batch(batch_size))
        return ret
        
    # define print step
    def print_step():
        beta_smoothing = print_loss_smoothing * (1 - (1-1/print_loss_smoothing)**(iteration+1))
        line = f'Iter {iteration:7d}'
        line += f' L {(state["loss_avg"] / beta_smoothing) ** 0.5:7.5f}'
        if 'atomic' not in model_type:
            line += f' LE {(state["le_avg"] / beta_smoothing) ** 0.5:7.5f}'
            line += f' LF {(state["lf_avg"] / beta_smoothing) ** 0.5:7.5f}'
        if model_type == 'atomic_iter':
            line += f' [{" ".join([f"{l:6.4f}" for l in (state["loss_avg_each_step"] / beta_smoothing)**0.5])}]'
        if use_val_data:
            if 'atomic' not in model_type:
                line += f' LEval {np.array([l[0] for l in loss_val]).mean() ** 0.5:7.5f}'
                line += f' LFval {np.array([l[1] for l in loss_val]).mean() ** 0.5:7.5f}'
            elif model_type == 'atomic':
                line += f' Lval {np.array(loss_val).mean() ** 0.5:7.5f}'
            elif model_type == 'atomic_iter':
                # line += f' [{" ".join([f"{l:6.4f}" for l in np.array(loss_val).mean(axis=0) ** 0.5])}]'
                line += f' [{" ".join([f"{l:6.4f}" for l in np.array(loss_val).max(axis=0) ** 0.5])}]'
        line += f' Time {time.time() - tic:.2f}s'
        print(line)

    # training loop
    tic = time.time()
    for iteration in range(int(step+1)):
        if use_val_data and iteration % print_every == 0:
            val_batch = get_batch_val()
            loss_val = []
            for one_batch in val_batch:
                v_batch, type_count, lattice_args = one_batch
                if model_type == 'atomic_iter':
                    v_batch, static_args = process_batch(v_batch)
                else:
                    static_args = nn.FrozenDict({'type_count': tuple(type_count),
                                                 'lattice': lattice_args})
                loss_val.append(val_step(v_batch, variables, static_args))
        batch, type_count, lattice_args = get_batch_train()
        if model_type == 'atomic_iter':
            batch, static_args = process_batch(batch)
        else:
            static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})
        variables, opt_state, state = train_step(batch,
                                                 variables,
                                                 opt_state,
                                                 state,
                                                 static_args)
        if iteration % print_every == 0:
            print_step()
            tic = time.time()

    # compress, save, and finish
    if compress and not model_type == 'atomic_iter':
        model, variables = compress_model(model,
                                        variables,
                                        compress_Ngrids,
                                        compress_r_min)
    save_model(save_path, model, variables)
    print(f'# Training finished in {datetime.timedelta(seconds=int(time.time() - TIC))}.')

def test(
    model_path: str,
    data_path: str,
    batch_size: int = 1,
    n_iter = None,
    wc_init = None,
    grad: bool = False,
    n_iter_grad: int = 1,
):
    '''
        Testing a trained model on a **single** dataset.
        Input arguments:
            model_path: path to the trained model.
            data_path: path to the data for evaluation.
            batch_size: Increase for potentially faster evaluation, but requires more memory.
    '''
    if type(data_path) == list:
        raise ValueError('Data_path should be a single string path for now.')

    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for evaluation.')
    
    model, variables = load_model(model_path, replicate=False)
    if model.params['type'] == 'energy':
        labels = ['coord', 'box', 'force', 'energy']
        atomic_sel = None
    elif model.params['type'] == 'atomic':
        labels = ['coord', 'box', model.params['atomic_data_prefix']]
        atomic_sel = model.params['nsel']
    elif model.params['type'] == 'atomic_iter':
        labels = ['coord', 'box', model.params['atomic_data_prefix']]
        atomic_sel = []
    elif 'dplr' in model.params['type']:
        labels = ['coord', 'box', 'force', 'energy']
        atomic_sel = None
    else:
        raise ValueError('Model type should be either "energy" or "atomic".')
    test_data = DPDataset([data_path],
                          labels,
                          {'atomic_sel':atomic_sel})
    test_data.compute_lattice_candidate(model.params['rcut'])
    if 'dplr' in model.params['type']:
        subsets = test_data.get_flattened_data()
        for subset in subsets:
            process_long_range_subset(subset,
                                      model.params['type'],
                                      model.params['dplr_q_atoms'],
                                      model.params['dplr_q_wc'],
                                      model.params['dplr_beta'],
                                      model.params.get('dplr_q1_beta', None),
                                      model.params['dplr_resolution'] * 0.6,
                                      *model.params['dplr_wannier_model_and_variables'])
    test_data.pointer = 0
    remaining = test_data.nframes
    if model.params['type'] == 'energy' or 'dplr' in model.params['type']:
        evaluate_fn = model.energy_and_force
        predictions = {'energy': [], 'force': []}
        ground_truth = {'energy': [], 'force': []}
    else:
        evaluate_fn = lambda variables, coord, box, static_args: model.apply(variables, coord, box, static_args)[0]
        predictions = {model.params['atomic_data_prefix']: []}
        ground_truth = {model.params['atomic_data_prefix']: []}
        if grad:
            predictions['jacobian'] = []
    if model.params['type'] == 'atomic_iter':
        def evaluate_fn_iter(variables, coord, box, static_args, wc_init, n_iter=n_iter):
            wc = wc_init
            for _ in range(n_iter):
                dwc, __ = model.apply(variables, jnp.concatenate([coord[0], wc]), box[0], static_args)
                wc = wc + dwc
            return wc[None]
        if grad:
            @partial(jax.jit, static_argnames=('static_args'))
            def evaluate_fn(variables, coord, box, static_args, wc_init):
                wc = evaluate_fn_iter(variables, coord, box, static_args, wc_init)[0]
                evaluate_fn_iter(variables, coord, box, static_args, wc, n_iter=n_iter_grad)
                jac = jax.jacrev(evaluate_fn_iter, argnums=1)(variables, coord, box, static_args, wc, n_iter=n_iter_grad)
                return wc[None], jac
        else:
            evaluate_fn = jax.jit(evaluate_fn_iter, static_argnames=('static_args', 'n_iter'))
    else:
        evaluate_fn = jax.jit(jax.vmap(evaluate_fn,
                                    in_axes=(None,0,0,None)),
                                    static_argnames=('static_args',))
    while remaining > 0:
        bs = min(batch_size, remaining) if not model.params['type'] == 'atomic_iter' else 1
        batch, type_count, lattice_args = test_data.get_batch(bs)
        remaining -= bs
        if model.params['type'] == 'atomic_iter':
            static_args = nn.FrozenDict({'type_count': tuple(type_count) + (batch["atomic"].shape[1],), 'lattice': lattice_args})
            pred = evaluate_fn(variables, batch['coord'], batch['box'], static_args, wc_init)
            if grad:
                pred, jac = pred
            wc_init = pred[0]
        else:
            static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})
            pred = evaluate_fn(variables, batch['coord'], batch['box'], static_args)
        if model.params['type'] == 'energy' or 'dplr' in model.params['type']:
            predictions['energy'].append(pred[0])
            predictions['force'].append(pred[1])
            ground_truth['energy'].append(batch['energy'])
            ground_truth['force'].append(batch['force'])
        else:
            if grad:
                predictions['jacobian'].append(jac)
            predictions[model.params['atomic_data_prefix']].append(pred)
            ground_truth[model.params['atomic_data_prefix']].append(batch["atomic"])
    rmse = {}
    for key in predictions.keys():
        predictions[key] = np.concatenate(predictions[key], axis=0)
        if key in ground_truth:
            ground_truth[key] = np.concatenate(ground_truth[key], axis=0)
            rmse[key] = (((predictions[key] - ground_truth[key]) ** 2).mean() ** 0.5).item()
    # reorder force back; will delete in future when reordering is moved into dpmodel.py
    if model.params['type'] == 'energy':
        ground_truth['force'] = ground_truth['force'][:, test_data.type.argsort(kind='stable').argsort(kind='stable')]
        predictions['force'] = predictions['force'][:, test_data.type.argsort(kind='stable').argsort(kind='stable')]
    return rmse, predictions, ground_truth
        
def evaluate(
    model_path: str,
    coord: np.ndarray,
    box: np.ndarray,
    type_idx: np.ndarray,
    batch_size: int = 1,
    init_wc = None,
    n_iter: int = None,
    grad: bool = False,
    n_iter_grad: int = 1,
):
    '''
        Evaluating a trained model on a set of configurations (without knowing ground truth).
        Input arguments:
            model_path: path to the trained model.
            coord: atomic coordinates of shape (n_frames, n_atoms, 3).
            box: simulation box of shape (n_frames) + (,) or (1,) or (3,) or (9), or (3,3).
            type_idx: atomic type indices of shape (Natoms,)
            batch_size: Increase for potentially faster evaluation, but requires more memory.
    '''
    # input shape check
    try:
        assert coord.ndim == 3 and coord.shape[2] == 3
        assert type_idx.ndim == 1 and box.ndim in [1, 2, 3]
        assert coord.shape[1] == type_idx.shape[0]
        assert coord.shape[0] == box.shape[0]
        if box.ndim == 1:
            box = box[:, None, None] * jnp.eye(3)
        elif box.ndim == 2:
            if box.shape[1] == 1:
                box = box[:, None] * jnp.eye(3)
            elif box.shape[1] == 3:
                box = jax.vmap(jnp.diag)(box)
            else:
                box = box.reshape(box.shape[0], 3, 3)
        elif box.ndim == 3:
            assert box.shape[1] == 3 and box.shape[2] == 3
    except:
        raise ValueError('Input shapes are incorrect: \n' + 
                         'coord: (n_frames, n_atoms, 3) \n' +
                         'box: (n_frames) + (,) or (1,) or (3,) or (9), or (3,3) \n' +
                         'type_idx (n_atoms).')
    
    model, _ = load_model(model_path, replicate=False)

    # create dataset in temp directory and use test() to evaluate
    with tempfile.TemporaryDirectory() as temp_dir:
        set_dir = os.path.join(temp_dir, "set.001")
        coord_path = os.path.join(set_dir, "coord.npy")
        box_path = os.path.join(set_dir, "box.npy")
        type_idx_path = os.path.join(temp_dir, "type.raw")
        os.makedirs(set_dir, exist_ok=True)
        np.save(coord_path, coord.reshape(coord.shape[0], -1))
        np.save(box_path, box.reshape(box.shape[0], -1))
        with open(type_idx_path, "w") as f:
            f.write("\n".join(np.array(type_idx, dtype=int).astype(str)))
        if model.params['type'] == 'atomic':
            atomic_path = os.path.join(set_dir, model.params['atomic_data_prefix'] + ".npy")
            label_count = np.isin(type_idx, model.params['nsel']).sum()
            np.save(atomic_path, np.zeros((coord.shape[0], label_count * 3)))
        elif model.params['type'] == 'energy':
            energy_path = os.path.join(set_dir, "energy.npy")
            force_path = os.path.join(set_dir, "force.npy")
            np.save(energy_path, np.zeros(coord.shape[0]))
            np.save(force_path, np.zeros((coord.reshape(coord.shape[0], -1)).shape))
        elif model.params['type'] == 'atomic_iter':
            if init_wc is None:
                raise ValueError("Must provide init_wc for 'atomic_iter' model type.")
            atomic_path = os.path.join(set_dir, f"{model.params['atomic_data_prefix']}.npy")
            np.save(atomic_path, np.zeros((coord.shape[0],) + init_wc.shape))
            _, predictions, _ = test(model_path, temp_dir, batch_size, n_iter=n_iter, wc_init=init_wc, grad=grad, n_iter_grad=n_iter_grad)
            return predictions[model.params['atomic_data_prefix']] if not grad else (predictions[model.params['atomic_data_prefix']], predictions['jacobian'])
        _, predictions, _ = test(model_path, temp_dir, batch_size)

    return predictions


def process_long_range_subset(subset, model_type, dplr_q_atoms, dplr_q_wc, dplr_beta, dplr_q1_beta, dplr_resolution, wc_model, wc_variables):
    '''
        subtracting long range energy and force, keeping short range part only, for dplr/dplr_q1 models.
    '''
    data, type_count, lattice_args = subset.values()
    if not lattice_args['ortho']:
        raise ValueError('For "dplr" currently only orthorhombic boxes are supported.')
    sel_type_count = tuple(np.array(type_count)[wc_model.params['nsel']])
    qatoms = np.repeat(dplr_q_atoms, type_count)
    qwc = np.repeat(dplr_q_wc, sel_type_count)
    static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})

    def lr_energy(coord, box, Ngrid):
        wc = wc_model.wc_predict(wc_variables, coord, box, static_args)
        p3mlr_fn = get_p3mlr_fn(jnp.diag(box), dplr_beta, Ngrid)
        return p3mlr_fn(jnp.concatenate([coord, wc]), jnp.concatenate([qatoms, qwc]))
    
    @partial(jax.jit, static_argnums=(2,))
    def lr_energy_and_force(coord, box, Ngrid):
        e, negf = jax.value_and_grad(lr_energy)(coord, box, Ngrid)
        return e, -negf
    
    if model_type == 'dplr_q1':
        def E_electron(coord, box, Ngrid):
            solve_E_electron = get_schrodinger_1e_fn(jnp.diag(box), dplr_q1_beta, wc_model, wc_variables, 
                                                     dplr_q_atoms, dplr_q_wc, static_args, M=Ngrid)[0]
            psi = jax.random.normal(jax.random.PRNGKey(0), Ngrid, jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32)
            psi = psi / jnp.linalg.norm(psi)
            return solve_E_electron(coord, box, None, psi)[0]
        @partial(jax.jit, static_argnums=(2,))
        def EF_electron(coord, box, Ngrid):
            e, negf = jax.value_and_grad(E_electron)(coord, box, Ngrid)
            return e, -negf

    for i in range(len(data['coord'])):
        if model_type == 'dplr_q1':
            Ngrid = get_p3mlr_grid_size(np.diag(data['box'][i]), max(dplr_beta, dplr_q1_beta), resolution=dplr_resolution)
        else:
            Ngrid = get_p3mlr_grid_size(np.diag(data['box'][i]), dplr_beta, resolution=dplr_resolution)
        e_lr, f_lr = lr_energy_and_force(data['coord'][i], data['box'][i], Ngrid)
        data['energy'][i] -= e_lr
        data['force'][i] -= f_lr
        if model_type == 'dplr_q1':
            e_se, f_se = EF_electron(data['coord'][i], data['box'][i], Ngrid)
            data['energy'][i] -= e_se
            data['force'][i] -= f_se