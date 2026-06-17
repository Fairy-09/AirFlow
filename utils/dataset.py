from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


class NormalizationRouter:
    def __init__(self, drift_threshold=0.45, acf_threshold=0.50, period_lag=24, smooth_window=24 * 7):
        self.drift_threshold = drift_threshold
        self.acf_threshold = acf_threshold
        self.period_lag = period_lag
        self.smooth_window = smooth_window

    def route(self, data_series, feature_name):
        feature_name = feature_name.lower()
        series_pd = pd.Series(data_series).dropna()

        if len(series_pd) < self.smooth_window * 2:
            return 'revin'

        rolling_mean = series_pd.rolling(window=self.smooth_window, min_periods=1).mean()
        global_std = np.std(data_series) + 1e-8
        trend_std = np.std(rolling_mean.dropna())
        drift_ratio = trend_std / global_std

        autocorr = series_pd.autocorr(lag=self.period_lag)
        if pd.isna(autocorr):
            autocorr = 0.0

        if autocorr >= self.acf_threshold:
            return 'minmax'
        elif drift_ratio >= self.drift_threshold:
            return 'revin'
        else:
            return 'minmax'


class AirQualityDataset(Dataset):
    def __init__(self, root, data_file, sequence_length=24, target_steps=24, num_stations=None,
                 target_pollutant='pm25'):
        self.root = root
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.target_steps = target_steps
        self.target_pollutant = target_pollutant.lower()

        self.router = NormalizationRouter()

        self.pollutant_mapping = {'pm25': ['pm25', 'PM2.5', 'PM25'], 'pm10': ['pm10', 'PM10'], 'so2': ['so2', 'SO2'],
                                  'no2': ['no2', 'NO2'], 'o3': ['o3', 'O3'], 'co': ['co', 'CO'], 'aqi': ['aqi', 'AQI']}
        if self.target_pollutant not in self.pollutant_mapping: raise ValueError(
            f"Unsupported target pollutant: {target_pollutant}.")
        self.data = pd.read_csv(os.path.join(self.root, self.data_file), encoding='gbk')
        if 'datetime' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
            self.data.drop('datetime', axis=1, inplace=True)
        elif 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        self.target_column = None
        for possible_name in self.pollutant_mapping[self.target_pollutant]:
            if possible_name in self.data.columns:
                self.target_column = possible_name
                break
        if self.target_column != self.target_pollutant:
            self.data[self.target_pollutant] = self.data[self.target_column]
            self.data.drop(self.target_column, axis=1, inplace=True)

        self.data[self.target_pollutant] = self.data[self.target_pollutant].fillna(method='ffill').fillna(
            method='bfill')
        self.data = self.data.sort_values(['station', 'timestamp']).reset_index(drop=True)

        available_stations = self.data['station'].unique()
        if num_stations is not None and num_stations < len(available_stations):
            self.data = self.data[self.data['station'].isin(available_stations[:num_stations])]

        self.data['hour'] = self.data['timestamp'].dt.hour / 23.0
        self.data['weekday'] = self.data['timestamp'].dt.dayofweek / 6.0
        self.time_cols = ['hour', 'weekday']

        available_columns = self.data.columns.tolist()
        all_pollutant_names = [name for names_list in self.pollutant_mapping.values() for name in names_list]
        continuous_cols = [self.target_pollutant] + [col for col in available_columns if
                                                     col in all_pollutant_names and col != self.target_pollutant]
        if self.target_pollutant != 'aqi' and 'aqi' in available_columns and 'aqi' not in continuous_cols:
            continuous_cols.append('aqi')

        self.revin_cols = []
        self.minmax_cols = []

        for col in continuous_cols:
            data_series = self.data[col].dropna().values
            method = self.router.route(data_series, col)
            if method == 'revin':
                self.revin_cols.append(col)
            else:
                self.minmax_cols.append(col)

        self.feature_cols = self.revin_cols + self.minmax_cols + self.time_cols
        self.num_revin_features = len(self.revin_cols)
        self.num_minmax_features = len(self.minmax_cols)

        self.target_in_revin = self.target_pollutant in self.revin_cols

        self.feature_scaler = None
        self.target_scaler = None

        if self.num_minmax_features > 0:
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.data[self.minmax_cols] = self.feature_scaler.fit_transform(self.data[self.minmax_cols].values)

        if not self.target_in_revin:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler.fit(self.data[self.target_pollutant].values.reshape(-1, 1))

        self.stations = self.data['station'].unique()
        self.train_data, self.val_data, self.test_data = [], [], []
        for station in self.stations:
            station_data = self.data[self.data['station'] == station].reset_index(drop=True)
            if len(station_data) < self.sequence_length + self.target_steps + 50: continue
            train_end, val_end = int(len(station_data) * 0.8), int(len(station_data) * 0.9)
            self.train_data.extend(self._create_sequences(station_data[:train_end].copy(), station))
            self.val_data.extend(self._create_sequences(station_data[train_end:val_end].copy(), station))
            self.test_data.extend(self._create_sequences(station_data[val_end:].copy(), station))

    def _create_sequences(self, station_data, station_name):
        sequences = []
        features = station_data[self.feature_cols].values
        target_values = station_data[self.target_pollutant].values
        for i in range(len(features) - self.sequence_length - self.target_steps + 1):
            input_seq = features[i:i + self.sequence_length]
            target_seq = target_values[i + self.sequence_length:i + self.sequence_length + self.target_steps]
            if not (np.isnan(input_seq).any() or np.isnan(target_seq).any() or np.isinf(input_seq).any() or np.isinf(
                    target_seq).any()):
                sequences.append((input_seq, target_seq, station_name))
        return sequences

    def get_train_data(self):
        return torch.stack([torch.from_numpy(seq[0]).float() for seq in self.train_data]), torch.stack(
            [torch.from_numpy(seq[1]).float() for seq in self.train_data])

    def get_val_data(self):
        return torch.stack(
            [torch.from_numpy(seq[0]).float() for seq in self.val_data]) if self.val_data else None, torch.stack(
            [torch.from_numpy(seq[1]).float() for seq in self.val_data]) if self.val_data else None

    def get_test_data(self):
        return torch.stack(
            [torch.from_numpy(seq[0]).float() for seq in self.test_data]) if self.test_data else None, torch.stack(
            [torch.from_numpy(seq[1]).float() for seq in self.test_data]) if self.test_data else None

    def inverse_transform_target(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if not self.target_in_revin and self.target_scaler is not None:
            original_shape = predictions.shape
            predictions_denorm = self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
            return predictions_denorm.reshape(original_shape)
        return np.asarray(predictions)

    def get_data_info(self):
        target_idx_in_revin = self.revin_cols.index(self.target_pollutant) if self.target_in_revin else 0

        return {
            'target_pollutant': self.target_pollutant,
            'num_revin_features': self.num_revin_features,
            'num_minmax_features': self.num_minmax_features,
            'target_in_revin': self.target_in_revin,
            'target_idx_in_revin': target_idx_in_revin
        }
