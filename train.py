import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import argparse
import os
import datetime
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import random
from tqdm import tqdm
from dataclasses import dataclass
import pickle

from model.state_space_stream import StateSpaceBlock
from model.dynamics_stream import DynamicsTimeStep
from utils.util import set_seed
from utils.dataset import AirQualityDataset


class GateMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=-1)
        gate_weight = self.gate(concat)
        return gate_weight * x1 + (1 - gate_weight) * x2


class StreamLayer(nn.Module):
    def __init__(self, config, dynamics_hidden_ratio=0.5):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.dynamics_hidden = max(16, int(config.d_model * dynamics_hidden_ratio))

        self.state_space_branch = StateSpaceBlock(config)
        self.state_space_norm = nn.LayerNorm(config.d_model)

        self.dynamics_branch = DynamicsTimeStep(
            input_size=config.d_model,
            hidden_size=self.dynamics_hidden,
            dropout=0.1
        )
        self.dynamics_proj = nn.Linear(self.dynamics_hidden, config.d_model)
        self.dynamics_norm = nn.LayerNorm(config.d_model)

        self.state_to_dynamics_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.dynamics_to_state_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.cross_fusion = GateMixer(config.d_model)

        self.final_fusion = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model, config.d_model)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        residual = x

        state_out = self.state_space_branch(x)
        state_out = self.state_space_norm(state_out)

        dynamics_hidden = torch.zeros(batch_size, self.dynamics_hidden, device=x.device)
        dynamics_outputs = []

        for t in range(seq_len):
            dynamics_hidden = self.dynamics_branch(x[:, t, :], dynamics_hidden)
            dynamics_outputs.append(dynamics_hidden)

        dynamics_out = torch.stack(dynamics_outputs, dim=1)
        dynamics_out = self.dynamics_proj(dynamics_out)
        dynamics_out = self.dynamics_norm(dynamics_out)

        state_enhanced, _ = self.state_to_dynamics_attn(
            state_out, dynamics_out, dynamics_out
        )
        dynamics_enhanced, _ = self.dynamics_to_state_attn(
            dynamics_out, state_out, state_out
        )
        cross_fusion = self.cross_fusion(state_enhanced, dynamics_enhanced)
        cross_fusion = self.attn_norm(cross_fusion)
        fusion_input_final = cross_fusion

        output = self.final_fusion(fusion_input_final)
        output = self.output_norm(output)
        output = output + residual

        return output


class StreamBlock(nn.Module):
    def __init__(self, config, dynamics_hidden_ratio=0.5):
        super().__init__()
        self.layer = StreamLayer(
            config,
            dynamics_hidden_ratio=dynamics_hidden_ratio
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        return self.layer(self.norm(x))


class AirFlow(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len=24, hidden_dim=32,
                 n_layers=3, dropout=0.2,
                 dynamics_hidden_ratio=0.5, use_revin=True,
                 target_feature_idx=0, is_no2_target=False):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_revin = use_revin and not is_no2_target
        self.target_feature_idx = target_feature_idx
        self.is_no2_target = is_no2_target

        if self.use_revin:
            self.revin = self._create_revin(in_dim, target_feature_idx)

        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        @dataclass
        class StreamConfig:
            d_model: int = hidden_dim
            d_inner: int = hidden_dim * 2
            n_layers: int = 1
            dt_rank: int = max(4, hidden_dim // 16)
            d_state: int = 16
            expand_factor: int = 2
            d_conv: int = 4
            dt_min: float = 0.001
            dt_max: float = 0.1
            dt_init: str = "random"
            dt_scale: float = 1.0
            dt_init_floor: float = 1e-4
            bias: bool = False
            conv_bias: bool = True
            pscan: bool = True

        config = StreamConfig()

        self.layers = nn.ModuleList([
            StreamBlock(
                config=config,
                dynamics_hidden_ratio=dynamics_hidden_ratio
            ) for _ in range(n_layers)
        ])

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, out_dim)
        )

        self._init_weights()

    def _create_revin(self, num_features, target_feature_idx):
        class RevIN(nn.Module):
            def __init__(self, num_features, eps=1e-5, affine=True, target_feature_idx=0):
                super().__init__()
                self.num_features = num_features
                self.eps = eps
                self.affine = affine
                self.target_feature_idx = target_feature_idx
                if self.affine:
                    self.affine_weight = nn.Parameter(torch.ones(num_features))
                    self.affine_bias = nn.Parameter(torch.zeros(num_features))

            def normalize(self, x):
                mean = torch.mean(x, dim=1, keepdim=True).detach()
                std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
                x_norm = (x - mean) / std
                if self.affine:
                    x_norm = x_norm * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
                target_mean = mean[:, :, self.target_feature_idx:self.target_feature_idx + 1]
                target_std = std[:, :, self.target_feature_idx:self.target_feature_idx + 1]
                return x_norm, target_mean, target_std

            def denormalize(self, x_norm, target_mean, target_std):
                if len(x_norm.shape) == 2:
                    target_mean = target_mean.squeeze(1)
                    target_std = target_std.squeeze(1)
                    if self.affine:
                        target_weight = self.affine_weight[self.target_feature_idx]
                        target_bias = self.affine_bias[self.target_feature_idx]
                        x_norm = (x_norm - target_bias) / target_weight
                    x = x_norm * target_std + target_mean
                else:
                    raise ValueError(f"Unexpected input shape: {x_norm.shape}")
                return x

        return RevIN(num_features, affine=True, target_feature_idx=target_feature_idx)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_revin:
            x_norm, target_mean, target_std = self.revin.normalize(x)
            x = x_norm

        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        x = self.temporal_pool(x).squeeze(-1)

        output = self.decoder(x)

        if self.use_revin:
            output = self.revin.denormalize(output, target_mean, target_std)

        return output


class Loss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.15, gamma=0.05, huber_delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)

    def forward(self, predictions, targets):
        huber_loss = self.huber(predictions, targets)
        mse_loss = self.mse(predictions, targets)
        l1_loss = self.l1(predictions, targets)

        gradient_penalty = torch.tensor(0.0, device=predictions.device)
        if predictions.size(1) > 1:
            pred_gradients = predictions[:, 1:] - predictions[:, :-1]
            target_gradients = targets[:, 1:] - targets[:, :-1]
            gradient_penalty = torch.mean((pred_gradients - target_gradients) ** 2)

        total_loss = (self.alpha * huber_loss +
                      self.beta * l1_loss +
                      self.gamma * gradient_penalty)
        return total_loss


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6, restore_best_weights=True,
                 monitor_overfitting=True, overfitting_threshold=1.15, warmup_epochs=5):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor_overfitting = monitor_overfitting
        self.overfitting_threshold = overfitting_threshold
        self.warmup_epochs = warmup_epochs
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.overfitting_counter = 0
        self.epoch = 0

    def __call__(self, val_loss, train_loss, model):
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(model)
            return False

        if self.monitor_overfitting and train_loss > 0:
            overfitting_ratio = val_loss / train_loss
            if overfitting_ratio > self.overfitting_threshold:
                self.overfitting_counter += 1
            else:
                self.overfitting_counter = max(0, self.overfitting_counter - 1)

            if self.overfitting_counter >= 8:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}


class Augmentor:
    def __init__(self, noise_std=0.005, mask_prob=0.03, time_shift_range=1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.time_shift_range = time_shift_range

    def add_gaussian_noise(self, x, strength=1.0):
        if random.random() < 0.7:
            noise_std = self.noise_std * strength * torch.std(x, dim=(1, 2), keepdim=True)
            noise = torch.randn_like(x) * noise_std
            return x + noise
        return x

    def random_mask(self, x, strength=1.0):
        if random.random() < self.mask_prob * strength:
            batch_size, seq_len, features = x.shape
            mask_features = random.randint(1, min(2, features))
            feature_indices = random.sample(range(features), mask_features)
            for feat_idx in feature_indices:
                mask_len = random.randint(1, 2)
                start_pos = random.randint(0, seq_len - mask_len)
                x[:, start_pos:start_pos + mask_len, feat_idx] = 0
        return x

    def time_shift(self, x, strength=1.0):
        if random.random() < 0.2 * strength:
            shift = random.randint(-self.time_shift_range, self.time_shift_range)
            if shift != 0:
                x = torch.roll(x, shift, dims=1)
        return x

    def augment(self, x, strength=0.5):
        x = self.add_gaussian_noise(x, strength)
        x = self.random_mask(x, strength * 0.5)
        x = self.time_shift(x, strength * 0.3)
        return x


def calc_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float('inf'), float('inf'), float('inf'), float('inf'), -float('inf')

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0

    return mse, rmse, mae, mape, r2


def validate(args, model, val_data, dataset, epoch=None, phase="VALIDATION"):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    criterion = Loss()
    x_val, y_val = val_data

    with torch.no_grad():
        if args.cuda:
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        batch_size = 128
        num_samples = x_val.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_x = x_val[i:end_idx]
            batch_y = y_val[i:end_idx]

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    predictions_orig = all_predictions.numpy()
    targets_orig = all_targets.numpy()

    if args.target_pollutant == 'no2' and hasattr(dataset, 'target_scaler') and dataset.target_scaler is not None:
        predictions_orig = dataset.inverse_transform_target(predictions_orig)
        targets_orig = dataset.inverse_transform_target(targets_orig)

    MSE_orig, RMSE_orig, MAE_orig, MAPE_orig, R2_orig = calc_metrics(
        targets_orig.flatten(), predictions_orig.flatten(), args.target_pollutant)

    avg_loss = total_loss / num_batches

    return avg_loss, (MSE_orig, RMSE_orig, MAE_orig, MAPE_orig, R2_orig), (predictions_orig, targets_orig)


def train(args, model):
    dataset = AirQualityDataset(
        args.root, args.data_file,
        sequence_length=args.seq_len, target_steps=args.out_dim,
        num_stations=args.num_stations,
        target_pollutant=args.target_pollutant
    )

    x_train, y_train = dataset.get_train_data()
    x_val, y_val = dataset.get_val_data()

    if x_train is None or y_train is None:
        return

    criterion = Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01
    )

    augmenter = Augmentor(noise_std=0.005, mask_prob=0.02, time_shift_range=1)
    early_stopping = EarlyStopping(
        patience=20, min_delta=1e-6, monitor_overfitting=True,
        overfitting_threshold=1.15, warmup_epochs=5
    )

    use_amp = args.cuda and hasattr(torch.cuda, 'amp')
    scaler = GradScaler() if use_amp else None

    norm_method = "MinMax" if args.target_pollutant == 'no2' else "RevIN"

    train_losses = []
    val_losses = []
    learning_rates = []
    best_metrics = None
    best_val_loss = float('inf')

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()

        batch_size = args.batch_size
        num_samples = x_train.size(0)
        epoch_losses = []

        augmentation_strength = max(0.2, 0.8 * (1 - epoch / args.epochs))

        indices = torch.randperm(num_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Training Epoch {epoch + 1}")
        for batch_idx in pbar:
            end_idx = min(batch_idx + batch_size, num_samples)
            batch_x = x_train_shuffled[batch_idx:end_idx]
            batch_y = y_train_shuffled[batch_idx:end_idx]

            if epoch < args.epochs * 0.7:
                batch_x = augmenter.augment(batch_x, strength=augmentation_strength)

            if args.cuda:
                batch_x = batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if x_val is not None and y_val is not None:
            val_loss, metrics, (predictions, targets) = validate(
                args, model, (x_val, y_val), dataset, epoch, phase="VALIDATION")
            val_losses.append(val_loss)

            if best_metrics is None or metrics[0] < best_metrics[0]:
                best_metrics = metrics

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics
                }, os.path.join('result', f'{args.wandb_name}_best_model.pth'))

            if early_stopping(val_loss, avg_train_loss, model):
                break

    total_time = time.time() - start_time

    x_test, y_test = dataset.get_test_data()
    test_metrics = None

    if x_test is not None and y_test is not None:
        best_model_path = os.path.join('result', f'{args.wandb_name}_best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_metrics, (test_pred, test_true) = validate(
            args, model, (x_test, y_test), dataset, phase="TEST")

        os.makedirs('result/test', exist_ok=True)
        np.save(os.path.join('result/test', f'{args.wandb_name}_test_predictions.npy'), test_pred)
        np.save(os.path.join('result/test', f'{args.wandb_name}_test_targets.npy'), test_true)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_metrics': best_metrics,
        'test_metrics': test_metrics,
        'total_epochs': len(train_losses),
        'total_time': total_time,
        'target_pollutant': args.target_pollutant,
        'fusion_type': 'gated',
        'normalization': norm_method,
        'model_config': {
            'hidden_dim': args.hidden,
            'n_layers': args.layer,
            'use_revin': args.use_revin
        }
    }

    with open(os.path.join('result', f'{args.wandb_name}_training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', default=True, help='CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--hidden', type=int, default=32, help='Dimension of representations')
    parser.add_argument('--layer', type=int, default=3, help='Num of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--root", default='data', type=str, help="Data root directory")
    parser.add_argument("--data_file", default='default.csv', type=str)
    parser.add_argument('--in_dim', type=int, default=9, help='Input dimension')
    parser.add_argument('--out_dim', type=int, default=6, help='Output sequence length')
    parser.add_argument('--seq_len', type=int, default=12, help='Input sequence length')
    parser.add_argument('--num_stations', type=int, default=50, help='Number of stations')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_revin', type=bool, default=True, help='Use RevIN normalization')
    parser.add_argument('--target_pollutant', type=str, default='pm25',
                        help='Target pollutant (pm25, pm10, no2, aqi)')

    args = parser.parse_args()

    args.wandb_name = f"airflow_{args.target_pollutant}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args.cuda = args.use_cuda and torch.cuda.is_available()

    os.makedirs('result', exist_ok=True)

    set_seed(args.seed, args.cuda)

    temp_dataset = AirQualityDataset(
        args.root, args.data_file,
        sequence_length=args.seq_len, target_steps=args.out_dim,
        num_stations=args.num_stations,
        target_pollutant=args.target_pollutant
    )
    target_feature_idx = temp_dataset.get_target_pollutant_index()
    is_no2_target = (args.target_pollutant.lower() == 'no2')

    model = AirFlow(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        n_layers=args.layer,
        dropout=args.dropout,
        dynamics_hidden_ratio=0.5,
        use_revin=args.use_revin,
        target_feature_idx=target_feature_idx,
        is_no2_target=is_no2_target
    )

    if args.cuda:
        model = model.cuda()

    trained_model, history = train(args, model)
