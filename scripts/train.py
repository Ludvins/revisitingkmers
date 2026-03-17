"""Unified training script for all k-mer embedding models.

Trains any registered embedder on a paired-reads CSV file (left_read,right_read per line,
no header — the same format as train_2m.csv).

Examples
--------
Outputs land in the same folder as the model (experiment-centric layout):

    runs/my_exp/
    ├── model.model    ← --output runs/my_exp/model.model
    ├── model.loss     ← per-epoch loss values
    └── config.json    ← all CLI args + timestamp + git commit (auto-saved)

NonLinear (bern loss, GPU):
    python scripts/train.py --model nonlinear \\
        --input data/train_2m.csv --k 4 --dim 256 \\
        --epoch 300 --lr 0.001 --loss bern \\
        --neg_sample_per_pos 200 --batch_size 10000 --max_read_num 100000 \\
        --device cuda --output runs/nonlinear/model.model

UncertainGen (2-phase, GPU):
    python scripts/train.py --model uncertaingen \\
        --input data/train_2m.csv --k 4 --dim 256 \\
        --mean_epochs 50 --var_epochs 20 --lr 0.01 --k_form adaptive --alpha 1.0 \\
        --neg_sample_per_pos 200 --batch_size 100000 --max_read_num 100000 \\
        --device cuda --output runs/uncertaingen/model.model

UncertainGen warm-started from a pretrained NonLinear model:
    python scripts/train.py --model uncertaingen \\
        --input data/train_2m.csv --k 4 --dim 256 \\
        --pretrained runs/nonlinear/model.model \\
        --mean_epochs 0 --var_epochs 20 --lr 0.01 \\
        --device cuda --output runs/uncertaingen_ws/model.model

PCL (vMF, GPU):
    python scripts/train.py --model pcl \\
        --input data/train_2m.csv --k 4 --dim 256 \\
        --n_phases 0 --n_batches_per_half_phase 10000 \\
        --lr 0.001 --neg_sample_per_pos 32 --batch_size 512 --max_read_num 100000 \\
        --kappa_mode implicit --n_mc_samples 8 --loss_kappa_init 16 \\
        --device cuda --output runs/pcl/model.model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import datetime
import subprocess
import sys


def _loss_path(output: str) -> str:
    """Return the canonical loss file path for a given model output path."""
    return os.path.join(os.path.dirname(os.path.abspath(output)), "model.loss")


def _save_config(args):
    """Save all CLI args + metadata to config.json in the model's output directory."""
    out_dir = os.path.dirname(os.path.abspath(args.output))
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    config = {
        **vars(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": git_hash,
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_nonlinear(args):
    from functools import partial
    from embedders.nonlinear import NonLinearEmbedder, contrastive_loss
    from datasets.paired_reads import PairedReadsDataset
    from utils.train import train_contrastive

    model = NonLinearEmbedder(
        k=args.k, dim=args.dim,
        device=args.device, verbose=True, seed=args.seed,
    )
    dataset = PairedReadsDataset(
        file_path=args.input,
        transform_func=model._feature_extractor.extract,
        neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num,
        seed=args.seed,
    )
    loss_fn = partial(contrastive_loss, scale=args.loss_scale,
                      neg_threshold=args.neg_threshold)
    loss = train_contrastive(
        model=model, dataset=dataset, loss_fn=loss_fn,
        lr=args.lr, epochs=args.epoch, device=args.device,
        batch_size=args.batch_size, num_workers=args.workers_num,
        loss_name=args.loss, save_path=args.output,
        loss_log_path=_loss_path(args.output),
        checkpoint_interval=args.checkpoint, verbose=True,
    )
    _save_loss(loss, _loss_path(args.output))
    print(f"Model saved to: {args.output}")


def train_uncertaingen(args):
    from functools import partial
    from embedders.uncertaingen import UncertainGenEmbedder, train_variance_phase
    from embedders.nonlinear import contrastive_loss
    from datasets.paired_reads import PairedReadsDataset
    from utils.train import train_contrastive

    model = UncertainGenEmbedder(
        k=args.k, dim=args.dim, alpha=args.alpha, k_form=args.k_form,
        device=args.device, verbose=True, seed=args.seed,
    )
    if args.pretrained:
        from embedders.nonlinear import NonLinearEmbedder
        source = NonLinearEmbedder.load(args.pretrained, device=args.device)
        model.copy_mean_from(source)
        print(f"Copied mean network from {args.pretrained}")

    dataset = PairedReadsDataset(
        file_path=args.input,
        transform_func=model._feature_extractor.extract,
        neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num,
        seed=args.seed,
    )

    if args.mean_epochs > 0:
        print("=" * 60)
        print("PHASE 1: Training mean network")
        print("=" * 60)
        p1_loss_fn = partial(contrastive_loss, scale=args.p1_scale)
        p1_loss = train_contrastive(
            model=model, dataset=dataset, loss_fn=p1_loss_fn,
            lr=args.lr, epochs=args.mean_epochs, device=args.device,
            batch_size=args.batch_size, num_workers=args.workers_num,
            loss_name="bern", parameters=model.mean_parameters(), verbose=True,
        )
        _save_loss(p1_loss, _loss_path(args.output).replace("model.loss", "mean.loss"))
    else:
        print("Skipping Phase 1 (mean_epochs=0, using pretrained mean network).")

    print("=" * 60)
    print("PHASE 2: Training variance network")
    print("=" * 60)
    p2_loss = train_variance_phase(
        model=model, dataset=dataset,
        lr=args.lr, epochs=args.var_epochs, device=args.device,
        batch_size=args.batch_size, num_workers=args.workers_num,
        alpha=args.alpha, neg_threshold=args.neg_threshold, verbose=True,
    )
    _save_loss(p2_loss, _loss_path(args.output).replace("model.loss", "variance.loss"))

    model.save(args.output)
    print(f"Model saved to: {args.output}")


def train_pcl(args):
    from embedders.pcl import PCLEmbedder, train_pcl as _train_pcl
    from datasets.paired_reads import PairedReadsDataset

    model = PCLEmbedder(
        k=args.k, dim=args.dim, device=args.device,
        verbose=True, seed=args.seed,
        kappa_mode=args.kappa_mode,
        kappa_min=args.kappa_min, kappa_max=args.kappa_max,
        kappa_init=args.kappa_init,
        hidden_dims=args.hidden_dims or None,
        activation=args.activation,
        use_batchnorm=not args.no_batchnorm,
        dropout=args.dropout,
        kappa_hidden_dims=args.kappa_hidden_dims or None,
    )

    if args.pretrained:
        from embedders.nonlinear import NonLinearEmbedder
        source = NonLinearEmbedder.load(args.pretrained, device=args.device)
        model.copy_mean_from(source)
        print(f"Copied mean network from {args.pretrained}")

    dataset = PairedReadsDataset(
        file_path=args.input,
        transform_func=model._feature_extractor.extract,
        neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num,
        seed=args.seed,
    )

    if args.kappa_mode == "explicit":
        import torch
        profiles = model._feature_extractor.extract_batch(
            [dataset._reads[i] for i in range(min(100, len(dataset._reads)))]
        )
        sample_features = torch.from_numpy(profiles).float().to(args.device)
        model._rescale_kappa(sample_features)
        print(f"Calibrated kappa: upscale={model.kappa_upscale.item():.4f}, "
              f"add={model.kappa_add.item():.4f}")

    losses = _train_pcl(
        model=model, dataset=dataset,
        lr=args.lr, device=args.device,
        batch_size=args.batch_size, num_workers=args.workers_num,
        n_phases=args.n_phases,
        n_batches_per_half_phase=args.n_batches_per_half_phase,
        n_mc_samples=args.n_mc_samples,
        loss_kappa_init=args.loss_kappa_init,
        learn_loss_kappa=not args.no_learn_loss_kappa,
        lr_decrease_after_phase=args.lr_decrease_after_phase,
        save_path=args.output, verbose=True,
    )
    _save_loss(losses, _loss_path(args.output))
    print(f"Model saved to: {args.output}")


def _save_loss(loss, path):
    with open(path, "w") as f:
        for v in loss:
            f.write(f"{v}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train a k-mer embedding model (unified entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument("--model", required=True,
                        choices=["nonlinear", "uncertaingen", "pcl"],
                        help="Model architecture to train.")
    parser.add_argument("--input", required=True,
                        help="Paired-reads CSV (left_read,right_read per line, no header).")
    parser.add_argument("--output", required=True,
                        help="Output path for the saved model (.model file).")

    # ── Shared hyperparameters ────────────────────────────────────────────
    parser.add_argument("--k", type=int, default=4, help="K-mer size.")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--epoch", type=int, default=300,
                        help="Training epochs (nonlinear). "
                             "UncertainGen uses --mean_epochs/--var_epochs instead.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size (0 = full dataset).")
    parser.add_argument("--neg_sample_per_pos", type=int, default=200,
                        help="Negative samples per positive pair.")
    parser.add_argument("--max_read_num", type=int, default=100000,
                        help="Maximum paired reads to load from CSV.")
    parser.add_argument("--workers_num", type=int, default=1,
                        help="DataLoader worker processes.")
    parser.add_argument("--device", default="cpu",
                        help="Compute device: 'cpu' or 'cuda'.")
    parser.add_argument("--seed", type=int, default=26042024,
                        help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Save checkpoint every N epochs (0 = disabled).")

    # ── NonLinear / Transformer ────────────────────────────────────────────
    parser.add_argument("--loss", default="bern",
                        choices=["bern", "poisson", "hinge"],
                        help="Contrastive loss function.")
    parser.add_argument("--loss_scale", type=float, default=1.0,
                        help="Scale for squared distance in loss: log_p = -scale * d².")
    parser.add_argument("--neg_threshold", type=float, default=None,
                        help="Negative-pair threshold for bern loss (None = standard BCE).")

    # ── UncertainGen ──────────────────────────────────────────────────────
    parser.add_argument("--mean_epochs", type=int, default=50,
                        help="[uncertaingen] Phase 1 epochs (mean network).")
    parser.add_argument("--var_epochs", type=int, default=20,
                        help="[uncertaingen] Phase 2 epochs (variance network).")
    parser.add_argument("--k_form", default="adaptive",
                        choices=["adaptive", "identity", "expected_distance"],
                        help="[uncertaingen] K-matrix form.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="[uncertaingen] Covariance scaling factor.")
    parser.add_argument("--p1_scale", type=float, default=1.0,
                        help="[uncertaingen] Phase 1 loss scale.")
    parser.add_argument("--pretrained", default=None,
                        help="[uncertaingen/pcl] Path to a pretrained NonLinearEmbedder "
                             "to initialise the mean network from. "
                             "Set --mean_epochs 0 to skip Phase 1 and go straight to variance training.")

    # ── PCL ───────────────────────────────────────────────────────────────
    parser.add_argument("--n_phases", type=int, default=0,
                        help="[pcl] Alternating training phases (0 = joint).")
    parser.add_argument("--n_batches_per_half_phase", type=int, default=10000,
                        help="[pcl] Batches per half-phase.")
    parser.add_argument("--lr_decrease_after_phase", type=float, default=0.5,
                        help="[pcl] LR decay factor after each half-phase.")
    parser.add_argument("--n_mc_samples", type=int, default=8,
                        help="[pcl] MC samples for MCInfoNCE loss.")
    parser.add_argument("--loss_kappa_init", type=float, default=16.0,
                        help="[pcl] Initial learnable loss temperature.")
    parser.add_argument("--no_learn_loss_kappa", action="store_true",
                        help="[pcl] Fix loss kappa (do not learn it).")
    parser.add_argument("--kappa_mode", default="implicit",
                        choices=["implicit", "explicit"],
                        help="[pcl] Kappa parameterization mode.")
    parser.add_argument("--kappa_min", type=float, default=16.0,
                        help="[pcl] Min kappa for explicit calibration.")
    parser.add_argument("--kappa_max", type=float, default=128.0,
                        help="[pcl] Max kappa for explicit calibration.")
    parser.add_argument("--kappa_init", type=float, default=32.0,
                        help="[pcl] Initial kappa for explicit mode.")
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        help="[pcl] Hidden layer sizes for mu network (default: [512]).")
    parser.add_argument("--activation", default="sigmoid",
                        choices=["sigmoid", "leaky_relu", "relu"],
                        help="[pcl] Activation function.")
    parser.add_argument("--no_batchnorm", action="store_true",
                        help="[pcl] Disable BatchNorm in mu/kappa networks.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="[pcl] Dropout probability.")
    parser.add_argument("--kappa_hidden_dims", type=int, nargs="+",
                        help="[pcl] Hidden dims for kappa network (default: same as mu).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _save_config(args)
    _set_seed(args.seed)

    dispatch = {
        "nonlinear": train_nonlinear,
        "uncertaingen": train_uncertaingen,
        "pcl": train_pcl,
    }
    dispatch[args.model](args)


if __name__ == "__main__":
    main()
