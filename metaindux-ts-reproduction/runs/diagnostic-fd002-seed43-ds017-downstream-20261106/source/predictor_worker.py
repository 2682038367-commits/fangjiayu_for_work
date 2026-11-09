def downstream(seed, rule):
    import numpy as np
    import pandas as pd
    import torch
    check_inputs()
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / 'upstream'))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        import eva_regressor as eva
        import wandb
    finally:
        sys.argv = argv
    # User requested equal training budgets. Keep the public loop and synthetic-dev
    # selection, removing only its epoch-end early-stop branch in this local worker.
    public_source = inspect.getsource(eva.train)
    early_stop = "        if epoch - best_epoch >= 10:\n            print('*******STOP!!!********')\n            break"
    assert public_source.count(early_stop) == 1
    fixed_source = public_source.replace(early_stop, '        # Fixed 35-epoch paired budget; early stopping disabled.')
    exec(compile(fixed_source, '<fd002_fixed_budget_public_predictor>', 'exec'), eva.__dict__)
    helper = module(DOWNSTREAM, 'frozen_original_predictor_fit')
    target = folder(seed, rule)
    helper.OUT = target
    helper.CHECKPOINTS = ROOT / f'checkpoints/evaluators/fd002_seed43_ds017_downstream_20261106/seed{seed}/{rule}'
    helper.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with np.load(target / f'samples/checkpoint{seed}_sample103.npz') as z:
        generated = dict(data=z['data'].copy(), label=z['label'].copy())
    with np.load(RUN / 'frozen_plan.npz') as z:
        ti, di = z['train_indices'], z['dev_indices']
    with np.load(RUN / 'real_validation.npz') as z:
        vx = torch.from_numpy(z['data'].copy())
        y, ids = z['label'].astype(float), z['engine_id'].copy()
        vy = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    args = SimpleNamespace(**check_inputs()['config']['predictor_args'])
    epoch = selected(seed, rule)['epoch_one_based']
    sy = generated['label'].ravel().astype(float)
    rows, perengine, audits = [], [], []
    wandb.init(project='fd002-final70-downstream', mode='disabled')
    try:
        for s in EVALUATORS:
            name = f'epoch{epoch}_eval{s}'
            with (target / 'logs' / f'{name}.log').open('w', buffering=1) as handle, contextlib.redirect_stdout(handle):
                audit, vp, sp = helper.fit(eva, args, generated, vx, vy, ti, di, epoch, s)
            assert audit['completed_epochs'] == 35 and audit['actual_updates'] == 3990
            audit.update(generator_seed=seed, rule=rule)
            audits.append(audit)
            meta = dict(generator_seed=seed, rule=rule, selected_epoch=epoch, evaluator_seed=s)
            for domain, p, yy, ei in [('real_validation', vp, y, ids),
                    ('synthetic_train', sp[ti], sy[ti], None), ('synthetic_dev', sp[di], sy[di], None)]:
                a, b = helper.metrics(p, yy, ei, meta, domain)
                rows.extend(a)
                perengine.extend(b)
            if rule == 'final70':
                best_epoch = selected(seed, 'best')['epoch_one_based']
                base = folder(seed, 'best')
                old = json.loads((base / 'audits' / f'epoch{best_epoch}_eval{s}.json').read_text())
                assert old['initialization_sha256'] == audit['initialization_sha256']
                with np.load(base / 'audits' / f'epoch{best_epoch}_eval{s}_orders.npz') as z:
                    old_order = z['orders']
                with np.load(target / 'audits' / f'{name}_orders.npz') as z:
                    new_order = z['orders']
                assert old_order.shape == new_order.shape == (35, 29077)
                assert np.array_equal(old_order, new_order), 'Unmatched predictor train order'
                assert old['actual_updates'] == audit['actual_updates'] == 3990
            write_csv(target / 'predictor_audits.csv', [{k: v for k, v in a.items() if k not in ['history', 'checkpoint_saves']} for a in audits])
            pd.DataFrame(rows).to_csv(target / 'per_model_segments.csv', index=False)
            pd.DataFrame(perengine).to_csv(target / 'per_engine.csv', index=False)
            dump(target / 'predictor_progress.json', dict(completed_fits=len(audits), total_fits=5, last_evaluator_seed=s, updated_at=now()))
            print('PREDICTOR', seed, rule, s, 'complete', flush=True)
    finally:
        wandb.finish()
    assert len(audits) == 5
