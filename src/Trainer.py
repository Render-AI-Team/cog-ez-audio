from cog import BaseTrainer, Input
import os
import sys
import subprocess

class Trainer(BaseTrainer):
    def train(
        self,
        config_name: str = Input(description="Path to config YAML", default="config/ezaudio-l.yml"),
        epochs: int = Input(description="Number of epochs", default=50),
        num_workers: int = Input(description="Number of data loader workers", default=16),
        num_threads: int = Input(description="Number of CPU threads", default=1),
        save_every_step: int = Input(description="Save every N steps", default=5000),
        random_seed: int = Input(description="Random seed", default=2024),
        log_step: int = Input(description="Log every N steps", default=100),
        log_dir: str = Input(description="Log directory", default="../logs/"),
        save_dir: str = Input(description="Checkpoint directory", default="../ckpts/"),
        ckpt: str = Input(description="Path to checkpoint for finetuning", default=None),
        strict: bool = Input(description="Strict checkpoint loading", default=False),
    ):
        """Launches training using src/train.py with provided arguments."""
        args = [
            sys.executable, "src/train.py",
            f"--config-name={config_name}",
            f"--epochs={epochs}",
            f"--num-workers={num_workers}",
            f"--num-threads={num_threads}",
            f"--save-every-step={save_every_step}",
            f"--random-seed={random_seed}",
            f"--log-step={log_step}",
            f"--log-dir={log_dir}",
            f"--save-dir={save_dir}",
        ]
        if ckpt:
            args.append(f"--ckpt={ckpt}")
        if strict:
            args.append(f"--strict={strict}")
        # Use accelerate if available
        if shutil.which("accelerate"):
            args = ["accelerate", "launch"] + args[1:]
        print("Running training command:", " ".join(args))
        result = subprocess.run(args, check=True)
        return result.returncode
