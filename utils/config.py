

class CFG:
    debug = False
    apex = False
    print_freq = 100
    num_workers = 8
    model_name = 'efficientnet_b3'

    size = 256
    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    scheduler = 'CosineAnnealingWarmRestarts'
    epochs = 10
    T_0 = 10  # CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 5
    target_col = 'label'
    n_fold = 5
    trn_fold = list(range(n_fold))  # [0, 1, 2, 3, 4]
    train = True
    inference = False


if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)
