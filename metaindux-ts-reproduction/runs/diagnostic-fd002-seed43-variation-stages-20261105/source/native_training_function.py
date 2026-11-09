def train(rep):
    import numpy as np
    import torch
    h = helper()
    h.OUT, h.WEIGHTS = OUT / rep, WEIGHTS / rep
    sys.path.insert(0, str(ROOT / 'upstream'))
    import utils
    original_seed = utils.torch_seed
    def seed(seed_value):
        original_seed(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        assert not torch.are_deterministic_algorithms_enabled()
    utils.torch_seed = seed
    original_step = torch.optim.Adam.step
    rows, epochs = [], []
    def digest_state(value):
        digest = hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                digest.update(str((v.dtype, tuple(v.shape))).encode())
                digest.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    digest.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v: visit(x)
            else:
                digest.update(repr(v).encode())
        visit(value)
        return digest.hexdigest()
    def step(opt, *a, **kw):
        actual = flags(torch)
        assert actual['benchmark'] is True and actual['deterministic'] is False and actual['global_strict'] is False
        frame = sys._getframe(1)
        while Path(frame.f_code.co_filename).name != 'TrainCondition.py':
            frame = frame.f_back
            assert frame is not None
        loss = float(frame.f_locals['loss'].detach().item())
        model = frame.f_locals['net_model']
        result = original_step(opt, *a, **kw)
        rows.append(dict(update=len(rows)+1, loss=loss))
        if len(rows) % 163 == 0:
            epochs.append(dict(epoch=len(rows)//163, updates=len(rows), model_sha256=h.tensor_hash(model.state_dict()),
                optimizer_sha256=digest_state(opt.state_dict()),
                cpu_rng_sha256=digest_state(torch.get_rng_state()), cuda_rng_sha256=digest_state(torch.cuda.get_rng_state()),
                numpy_rng_sha256=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(), flags=actual))
            save(h.OUT / 'seed43/epoch_state_hashes.json', epochs)
            print('NATIVE_TRAIN', rep, 'epoch', len(rows)//163, 'of70', flush=True)
        return result
    torch.optim.Adam.step = step
    try:
        h.train(43)
    finally:
        torch.optim.Adam.step = original_step
        utils.torch_seed = original_seed
        if rows:
            h.write_csv(h.OUT / 'seed43/update_losses.csv', rows)
    assert len(rows) == 11410 and len(epochs) == 70
    auditpath = h.OUT / 'seed43/training_audit.json'
    audit = json.loads(auditpath.read_text())
    audit.update(training_convolution_flags=flags(torch),
                 caveat='Fixed convolution selection; global strict mode remains disabled. Full-repeat claim depends on independent repeat comparison.')
    save(auditpath, audit)
