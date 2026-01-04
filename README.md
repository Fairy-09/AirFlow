# 🔥 AirFlow: Multi-Scale Dual-Stream Dynamics Modeling for Multivariate Air Quality Prediction

# 🌟 Overview

AirFlow is a dual-stream deep learning model for multivariate air quality forecasting. It combines a State Space Stream for capturing long-term dependencies with a Dynamics Stream for modeling short-term fluctuations, fused through cross-attention and learnable gating mechanisms.

# 📁 Project Structure

```
AirFlow/
├── model/
│   ├── dynamics_stream.py      # Dynamics Stream with liquid time-constant networks
│   ├── pscan.py                # Parallel scan algorithm for SSM
│   └── state_space_stream.py   # State Space Stream with selective SSM
├── utils/
│   ├── dataset.py              # Data loading and preprocessing
│   ├── logger.py               # Training logger
│   └── util.py                 # Utility functions (seed setting, etc.)
├── train.py                    # Main training script
```

# 📚 Requirements

- Python 3.8.19
- PyTorch 2.2.2
- CUDA 12.1

Install dependencies:

```bash
pip install -r requirements.txt
```

# 📊 Experimental Highlights

We use real-world air quality datasets from Beijing and Tianjin, China.

### Data Source

Download from: **https://quotsoft.net/air/**

### Baselines

- **RNN-based**: BiLSTM, ConvLSTM
- **Transformer-based**: Transformer, iTransformer, PatchTST, Airformer
- **GNN-based**: STAAGCN
- **Continuous-time-based**: Mamba, LNN

### Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination

# 🛠 Usage

### Training

Basic training command:

```bash
python train.py --target_pollutant pm25 --data_file beijing_air_quality.csv
```

### Key Arguments

| Argument             | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `--epochs`           | 60      | Number of training epochs            |
| `--lr`               | 2e-4    | Learning rate                        |
| `--batch_size`       | 32      | Batch size                           |
| `--hidden`           | 32      | Hidden dimension                     |
| `--layer`            | 3       | Number of HDS layers                 |
| `--seq_len`          | 12      | Input sequence length (hours)        |
| `--out_dim`          | 6       | Prediction horizon (hours)           |
| `--target_pollutant` | pm25    | Target pollutant (pm25/pm10/no2/aqi) |
| `--dropout`          | 0.1     | Dropout rate                         |
| `--use_revin`        | True    | Use RevIN normalization              |
| `--root`             | data    | Data directory                       |

### Example Commands

```bash
# Train PM2.5 prediction model
python train.py --target_pollutant pm25 --epochs 60 --batch_size 128

# Train PM10 prediction model with custom settings
python train.py --target_pollutant pm10 --hidden 64 --layer 4

# Train NO2 prediction model (uses MinMax normalization automatically)
python train.py --target_pollutant no2 --lr 1e-4

# Train AQI prediction model
python train.py --target_pollutant aqi --seq_len 24 --out_dim 12
```

# 🌟 Citation

If you find our repo useful for your research, please consider giving a 🌟 and citing our work below.

```
Under Review
```

# 🔐 License

This source code is provided for **research and education purposes only**. Any commercial use requires formal permission from the authors.

# 📩 Contact

If you have any questions, please open an issue or contact:

- Email: []
