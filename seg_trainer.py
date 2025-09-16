import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

class SegmentationTrainer:
    def __init__(self, model, train_loader, test_loader, args,
                 num_parts: int, part_names=None, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.args = args

        self.num_parts = int(num_parts)
        self.part_names = part_names if part_names is not None else {}

        self.categories = {}
        try:
            ds = getattr(train_loader, "dataset", None)
            self.categories = getattr(ds, "cat_to_idx", {}) or {}
        except Exception:
            pass

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)
        self.criterion = nn.NLLLoss()

        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_ious = []
        self.val_accuracies = []
        self.val_ious = []

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Optional TensorBoard
        self.tb = None
        if getattr(self.args, "use_tensorboard", False):
            try:
                run_name = f"seg_{self.args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
                tb_dir = os.path.join(self.args.checkpoint_dir, "runs", run_name)
                os.makedirs(tb_dir, exist_ok=True)
                print(f"[TensorBoard] Logging to: {tb_dir}")
                self.tb = SummaryWriter(log_dir=tb_dir)
            except Exception as e:
                print(f"[TensorBoard] Failed to init SummaryWriter: {e}")
                self.tb = None


    def _get_device(self):
        if self.args.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.args.device)

    def _feature_transform_regularizer(self, trans):
        d = trans.size(1)
        I = torch.eye(d, device=trans.device)[None, :, :]
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss

    def _ensure_bn_layout(self, seg_pred):
        if seg_pred.dim() != 3:
            raise RuntimeError(f"seg_pred should be 3D, got shape {tuple(seg_pred.shape)}")
        B, D1, D2 = seg_pred.shape
        if D1 == self.num_parts and D2 != self.num_parts:
            seg_pred = seg_pred.transpose(1, 2).contiguous()  # [B, N, num_parts]
        return seg_pred

    def _save_curves(self, save_path: str):
        """Save/update a PNG of curves during training (loss/acc/mIoU)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.train_losses) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

            # Left: Loss & Accuracy
            ax1.plot(epochs, self.train_losses, label="Train Loss")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True, alpha=0.3)

            ax1b = ax1.twinx()
            ax1b.plot(epochs, self.train_accuracies, label="Train Acc", linestyle="--")
            ax1b.plot(epochs, self.test_accuracies,  label="Test Acc",  linestyle="-.")
            ax1b.set_ylabel("Accuracy (%)")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="lower right")

            # Right: mIoU
            if self.test_ious:
                ax2.plot(epochs[:len(self.test_ious)], self.test_ious, label="Test mIoU (%)")
            if self.val_ious:
                ax2.plot(epochs[:len(self.val_ious)],  self.val_ious,  label="Val mIoU (%)", linestyle="--")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU (%)"); ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower right")

            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[Warn] Failed to save segmentation curves to {save_path}: {e}")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct_points = 0
        total_points = 0

        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")
        for data, seg_labels, cat_labels in train_pbar:
            data = data.to(self.device)            # [B,C,N]
            seg_labels = seg_labels.to(self.device)  # [B,N]
            cat_labels = cat_labels.to(self.device)  # [B]

            self.optimizer.zero_grad()
            seg_pred, trans, trans_feat = self.model(data, cat_labels)  # [B,N,k] or [B,k,N]
            seg_pred = self._ensure_bn_layout(seg_pred)

            loss = self.criterion(seg_pred.view(-1, self.num_parts),
                                  seg_labels.view(-1))
            if trans_feat is not None:
                loss = loss + 0.001 * self._feature_transform_regularizer(trans_feat)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pred_choice = seg_pred.argmax(dim=-1)  # [B,N]
            correct_points += pred_choice.eq(seg_labels).sum().item()
            total_points += seg_labels.numel()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc':  f'{100. * correct_points / max(1,total_points):.2f}%'
            })

        train_loss = running_loss / max(1, len(self.train_loader))
        train_acc = 100. * correct_points / max(1, total_points)
        return train_loss, train_acc

    def test_epoch(self, epoch, data_loader, mode='Test'):
        self.model.eval()
        correct_points = 0
        total_points = 0
        total_per_cat_iou = []

        with torch.no_grad():
            test_pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} - {mode}')
            for data, seg_labels, cat_labels in test_pbar:
                data = data.to(self.device)
                seg_labels = seg_labels.to(self.device)
                cat_labels = cat_labels.to(self.device)

                seg_pred, _, _ = self.model(data, cat_labels)
                seg_pred = self._ensure_bn_layout(seg_pred)

                pred_choice = seg_pred.argmax(dim=-1)  # [B,N]
                correct_points += pred_choice.eq(seg_labels).sum().item()
                total_points += seg_labels.numel()

                B, N = pred_choice.shape
                for i in range(B):
                    segp = pred_choice[i].detach().cpu().numpy()
                    segl = seg_labels[i].detach().cpu().numpy()
                    parts = np.unique(segl)
                    if parts.size == 0:
                        continue
                    ious = []
                    for p in parts:
                        if p < 0:  # skip invalid
                            continue
                        I = np.sum((segp == p) & (segl == p))
                        U = np.sum((segp == p) | (segl == p))
                        ious.append(1.0 if U == 0 else I / U)
                    if ious:
                        total_per_cat_iou.append(float(np.mean(ious)))

                miou = (np.mean(total_per_cat_iou) * 100) if total_per_cat_iou else 0.0
                test_pbar.set_postfix({
                    'Acc':  f'{100. * correct_points / max(1,total_points):.2f}%',
                    'mIoU': f'{miou:.2f}%'
                })

        test_acc = 100. * correct_points / max(1, total_points)
        mean_iou = (np.mean(total_per_cat_iou) * 100) if total_per_cat_iou else 0.0
        return test_acc, mean_iou

    def save_checkpoint(self, epoch, train_loss, test_acc, test_iou):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_iou': test_iou,
            'args': self.args,
            'num_parts': self.num_parts,
            'part_names': self.part_names,
        }
        path = os.path.join(self.args.checkpoint_dir, f'{self.args.model}_seg_epoch_{epoch+1}.pth')
        torch.save(ckpt, path)

    def save_final_model(self):
        final_path = os.path.join(self.args.checkpoint_dir, f'{self.args.model}_segmentation_final.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_parts': self.num_parts,
            'part_names': self.part_names,
            'args': self.args
        }, final_path)

    def train(self):
        print(f"Training segmentation model on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        if isinstance(self.categories, dict) and self.categories:
            print(f"Object categories: {len(self.categories)}")
        print(f"Model: {self.args.model}")

        curves_path = os.path.join(self.args.checkpoint_dir, f"{self.args.model}_seg_curves.png")

        best_test_acc = 0.0
        best_test_iou = 0.0

        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc, test_iou = self.test_epoch(epoch, self.test_loader, 'Test')

            if self.val_loader:
                val_acc, val_iou = self.test_epoch(epoch, self.val_loader, 'Validation')
                self.val_accuracies.append(val_acc)
                self.val_ious.append(val_iou)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.test_ious.append(test_iou)

            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Acc: {test_acc:.2f}%, Test mIoU: {test_iou:.2f}%')

            # TensorBoard scalars (optional)
            if self.tb is not None:
                self.tb.add_scalar("Loss/train", train_loss, epoch+1)
                self.tb.add_scalar("Acc/train",  train_acc,  epoch+1)
                self.tb.add_scalar("Acc/test",   test_acc,   epoch+1)
                self.tb.add_scalar("mIoU/test",  test_iou,   epoch+1)
                lr = self.optimizer.param_groups[0]['lr']
                self.tb.add_scalar("LR", lr, epoch+1)
                if self.val_loader:
                    self.tb.add_scalar("Acc/val",  val_acc, epoch+1)
                    self.tb.add_scalar("mIoU/val", val_iou, epoch+1)

            # Save/update the PNG plot every epoch
            self._save_curves(curves_path)

            best_test_acc = max(best_test_acc, test_acc)
            best_test_iou = max(best_test_iou, test_iou)

            self.scheduler.step()

            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss, test_acc, test_iou)

        self.save_final_model()

        print("Training completed!")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        print(f"Best Test mIoU: {best_test_iou:.2f}%")

        return self.train_losses, self.train_accuracies, self.test_accuracies, self.test_ious
