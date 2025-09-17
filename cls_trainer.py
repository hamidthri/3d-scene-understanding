import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Classification trainer with:
      - TensorBoard logging (enable via --use_tensorboard)
      - Checkpoints every --save_interval epochs
      - Final model snapshot
      - Optional PNG curve export at the end
    Expects the model to output log-probs for nn.NLLLoss() or logits for CrossEntropyLoss().
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, args, classes: List[str]):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.classes = classes

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # Optimizer / scheduler / loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        # If your model returns logits, you may prefer CrossEntropyLoss().
        # Your original code used NLLLoss(), so keep it unless you change the model head.
        self.criterion = nn.NLLLoss()

        # History buffers
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_accuracies: List[float] = []

        # Filesystem
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # ---- TensorBoard (optional, enabled with --use_tensorboard) ----
        self.tb = None
        if getattr(self.args, "use_tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_name = f"cls_{self.args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
                tb_dir = os.path.join(self.args.checkpoint_dir, "runs", run_name)
                os.makedirs(tb_dir, exist_ok=True)
                print(f"[TensorBoard] Logging to: {os.path.abspath(tb_dir)}")
                self.tb = SummaryWriter(log_dir=tb_dir)
            except Exception as e:
                print(f"[TensorBoard] Failed to init SummaryWriter: {e}")
                self.tb = None

    def _get_device(self) -> torch.device:
        if getattr(self.args, "device", "auto") == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        return torch.device(self.args.device)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")
        for data, target in train_pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Model should return (log_probs, trans, trans_feat) or just (log_probs, _, _)
            pred, trans, trans_feat = self.model(data)

            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            pred_choice = pred.max(1)[1]
            correct_train += int(pred_choice.eq(target).sum().item())
            total_train += int(target.size(0))

            train_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.0 * correct_train / max(1, total_train):.2f}%"
            })

        train_loss = running_loss / max(1, len(self.train_loader))
        train_acc = 100.0 * correct_train / max(1, total_train)
        return train_loss, train_acc

    def test_epoch(self, epoch: int) -> float:
        self.model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            test_pbar = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Testing")
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                pred, _, _ = self.model(data)

                pred_choice = pred.max(1)[1]
                correct_test += int(pred_choice.eq(target).sum().item())
                total_test += int(target.size(0))

                test_pbar.set_postfix({
                    "Acc": f"{100.0 * correct_test / max(1, total_test):.2f}%"
                })

        test_acc = 100.0 * correct_test / max(1, total_test)
        return test_acc

    def save_checkpoint(self, epoch: int, train_loss: float, test_acc: float) -> None:
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f"{self.args.model}_cls_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "test_acc": test_acc,
            "args": vars(self.args),
            "classes": self.classes,
        }, checkpoint_path)

    def save_final_model(self) -> None:
        final_path = os.path.join(self.args.checkpoint_dir, f"{self.args.model}_classification_final.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "classes": self.classes,
            "num_points": getattr(self.args, "num_points", None),
            "feature_transform": getattr(self.args, "feature_transform", None),
            "args": vars(self.args),
        }, final_path)

    def _save_png_curves(self) -> None:
        """Optional: save PNG curves (loss/acc) to checkpoint_dir."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(self.train_losses) + 1))
            out_png = os.path.join(self.args.checkpoint_dir, f"{self.args.model}_cls_curves.png")

            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(epochs, self.train_losses, label="Train Loss")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(epochs, self.train_accuracies, label="Train Acc", linestyle="--")
            ax2.plot(epochs, self.test_accuracies, label="Test Acc", linestyle="-.")
            ax2.set_ylabel("Accuracy (%)")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="lower right")

            plt.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[Curves] Saved {out_png}")
        except Exception as e:
            print(f"[Curves] Failed to save curves: {e}")

    def train(self) -> Tuple[List[float], List[float], List[float]]:
        print(f"Training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

        last_test_acc = 0.0

        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.test_epoch(epoch)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            # Print summary
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")

            # TensorBoard scalars
            if self.tb is not None:
                self.tb.add_scalar("train/loss", train_loss, epoch + 1)
                self.tb.add_scalar("train/acc", train_acc, epoch + 1)
                self.tb.add_scalar("test/acc", test_acc, epoch + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                self.tb.add_scalar("lr", lr, epoch + 1)
                self.tb.flush()

            # Step LR
            self.scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss, test_acc)

            last_test_acc = test_acc

        # Final artifacts
        self.save_final_model()
        self._save_png_curves()

        # Close TB
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()

        print(f"Final Test Accuracy: {last_test_acc:.2f}%")
        print("Training completed!")

        return self.train_losses, self.train_accuracies, self.test_accuracies
