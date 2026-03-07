"""
Command-line interface for dataset generation, CNN pretraining, and RL training.

Usage:
    python -m training.cli generate-dataset --output-dir data/train --n-samples 500
    python -m training.cli pretrain --dataset data/train --save pretrained.pt
    python -m training.cli train --dataset data/train --pretrained pretrained.pt -n 1000
    python -m training.cli train --n-episodes 200  # on-the-fly generation
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def cmd_generate_dataset(args):
    from training.rl_bandit_agent import precompute_dataset

    grid_sizes = [int(x) for x in args.grid_sizes.split(',')]
    print(f"Generating dataset: {args.n_samples} samples, "
          f"grids={grid_sizes}, seed={args.seed}")
    print(f"Output: {args.output_dir}")

    out = precompute_dataset(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        grid_sizes=grid_sizes,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(f"\nDataset saved to {out}")


def cmd_pretrain(args):
    import torch
    from training.rl_bandit_agent import (
        BanditAgent, CNNPretrainer, PrecomputedDataset, _get_device,
    )

    device = torch.device(args.device) if args.device else _get_device()
    print(f"Using device: {device}")

    agent = BanditAgent(D=args.embed_dim, n_transformer_blocks=args.n_blocks)

    n_cnn_params = sum(p.numel() for p in agent.patch_cnn.parameters())
    n_total = sum(p.numel() for p in agent.parameters())
    print(f"Agent: D={args.embed_dim}, {args.n_blocks} blocks, "
          f"{n_total:,} total params, {n_cnn_params:,} CNN params")

    dataset = None
    if args.dataset:
        dataset = PrecomputedDataset(args.dataset)
        print(f"Dataset: {len(dataset)} samples from {args.dataset}")

    pretrainer = CNNPretrainer(
        agent, lr=args.lr, log_dir=args.log_dir, device=device,
    )
    pretrainer.train(
        n_epochs=args.n_epochs,
        samples_per_epoch=args.samples_per_epoch,
        verbose=not args.quiet,
        dataset=dataset,
    )

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        # Save weights on CPU for portability
        torch.save({
            'agent_state_dict': {k: v.cpu() for k, v in agent.state_dict().items()},
            'pretrain_history': pretrainer.history,
            'args': vars(args),
        }, args.save)
        print(f"Pretrained checkpoint saved to {args.save}")


def cmd_train(args):
    import torch
    from training.rl_bandit_agent import (
        BanditAgent, BanditTrainer, PrecomputedDataset, _get_device,
    )

    device = torch.device(args.device) if args.device else _get_device()
    print(f"Using device: {device}")

    # Load or create agent
    agent = BanditAgent(D=args.embed_dim, n_transformer_blocks=args.n_blocks)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        # Load only matching keys (the CNN weights) from the pretrained checkpoint
        pretrained_state = checkpoint['agent_state_dict']
        agent_state = agent.state_dict()
        loaded_keys = []
        for k, v in pretrained_state.items():
            if k in agent_state and agent_state[k].shape == v.shape:
                agent_state[k] = v
                loaded_keys.append(k)
        agent.load_state_dict(agent_state)
        cnn_keys = [k for k in loaded_keys if k.startswith('patch_cnn.')]
        print(f"Loaded pretrained weights: {len(loaded_keys)} keys "
              f"({len(cnn_keys)} CNN params)")

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: D={args.embed_dim}, {args.n_blocks} blocks, {n_params:,} params")

    # Load precomputed dataset if provided
    dataset = None
    if args.dataset:
        dataset = PrecomputedDataset(args.dataset)
        print(f"Dataset: {len(dataset)} samples from {args.dataset}")

    trainer = BanditTrainer(
        agent,
        lambda_cost=args.lambda_cost,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        log_dir=args.log_dir,
        device=device,
    )

    # Restore optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    trainer.train(
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        dataset=dataset,
    )

    # Save checkpoint (weights on CPU for portability)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({
            'agent_state_dict': {k: v.cpu() for k, v in agent.state_dict().items()},
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'history': trainer.history,
            'args': vars(args),
        }, args.save)
        print(f"Checkpoint saved to {args.save}")

    # Save training history as JSON
    if args.history:
        os.makedirs(os.path.dirname(args.history) or '.', exist_ok=True)
        with open(args.history, 'w') as f:
            json.dump(trainer.history, f, indent=2)
        print(f"Training history saved to {args.history}")


def main():
    parser = argparse.ArgumentParser(
        prog='srw-train',
        description='SRW Parameter Advisor: dataset generation, pretraining, and training',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # --- generate-dataset ---
    gen = sub.add_parser('generate-dataset', help='Precompute training dataset')
    gen.add_argument('--output-dir', '-o', required=True,
                     help='Directory to save the dataset')
    gen.add_argument('--n-samples', '-n', type=int, default=500,
                     help='Number of samples to generate (default: 500)')
    gen.add_argument('--grid-sizes', default='128,256',
                     help='Comma-separated grid sizes (default: 128,256)')
    gen.add_argument('--seed', type=int, default=42,
                     help='Random seed (default: 42)')
    gen.add_argument('--quiet', '-q', action='store_true',
                     help='Suppress progress output')

    # --- pretrain ---
    pt = sub.add_parser('pretrain',
                        help='Pretrain CNN encoder on patch feature prediction')
    pt.add_argument('--dataset', '-d',
                    help='Path to precomputed dataset directory')
    pt.add_argument('--n-epochs', '-n', type=int, default=50,
                    help='Number of pretraining epochs (default: 50)')
    pt.add_argument('--samples-per-epoch', type=int, default=64,
                    help='Wavefronts per epoch (default: 64)')
    pt.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')
    pt.add_argument('--embed-dim', type=int, default=256,
                    help='Agent embedding dimension (default: 256)')
    pt.add_argument('--n-blocks', type=int, default=2,
                    help='Number of transformer blocks (default: 2)')
    pt.add_argument('--save', '-s',
                    help='Path to save pretrained checkpoint (.pt)')
    pt.add_argument('--log-dir',
                    help='TensorBoard log directory')
    pt.add_argument('--device',
                    help='Torch device (e.g. cuda, cpu). Default: auto')
    pt.add_argument('--quiet', '-q', action='store_true',
                    help='Suppress progress output')

    # --- train ---
    tr = sub.add_parser('train', help='Train the bandit agent')
    tr.add_argument('--dataset', '-d',
                    help='Path to precomputed dataset directory')
    tr.add_argument('--n-episodes', '-n', type=int, default=200,
                    help='Number of training episodes (default: 200)')
    tr.add_argument('--batch-size', '-b', type=int, default=8,
                    help='Episodes per gradient update (default: 8)')
    tr.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate (default: 3e-4)')
    tr.add_argument('--lambda-cost', type=float, default=0.1,
                    help='Cost penalty weight (default: 0.1)')
    tr.add_argument('--entropy-coeff', type=float, default=0.01,
                    help='Entropy bonus coefficient (default: 0.01)')
    tr.add_argument('--embed-dim', type=int, default=256,
                    help='Agent embedding dimension (default: 256)')
    tr.add_argument('--n-blocks', type=int, default=2,
                    help='Number of transformer blocks (default: 2)')
    tr.add_argument('--save', '-s',
                    help='Path to save model checkpoint (.pt)')
    tr.add_argument('--resume', '-r',
                    help='Path to checkpoint to resume from (.pt)')
    tr.add_argument('--pretrained',
                    help='Path to pretrained CNN checkpoint (.pt) for weight init')
    tr.add_argument('--history',
                    help='Path to save training history (.json)')
    tr.add_argument('--log-dir',
                    help='TensorBoard log directory')
    tr.add_argument('--device',
                    help='Torch device (e.g. cuda, cpu). Default: auto')
    tr.add_argument('--quiet', '-q', action='store_true',
                    help='Suppress progress output')

    args = parser.parse_args()

    if args.command == 'generate-dataset':
        cmd_generate_dataset(args)
    elif args.command == 'pretrain':
        cmd_pretrain(args)
    elif args.command == 'train':
        cmd_train(args)


if __name__ == '__main__':
    main()
