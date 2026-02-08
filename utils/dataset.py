from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


class AirQualityDataset(Dataset):
    def __init__(self, root, data_file, sequence_length=24, target_steps=24, num_stations=None,
                 target_pollutant='pm25'):
        self.root = root
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.target_steps = target_steps
        self.target_pollutant = target_pollutant.lower()
        self.is_no2_target = (self.target_pollutant == 'no2')

        self.pollutant_mapping = {
            'pm25': ['pm25', 'PM2.5', 'PM25'],
            'pm10': ['pm10', 'PM10'],
            'so2': ['so2', 'SO2'],
            'no2': ['no2', 'NO2'],
            'o3': ['o3', 'O3'],
            'co': ['co', 'CO'],
            'aqi': ['aqi', 'AQI']
        }

        if self.target_pollutant not in self.pollutant_mapping:
            raise ValueError(f"Unsupported target pollutant: {target_pollutant}. "
                             f"Supported: {list(self.pollutant_mapping.keys())}")

        self.data = pd.read_csv(os.path.join(self.root, self.data_file), encoding='gbk')

        if 'datetime' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
            self.data.drop('datetime', axis=1, inplace=True)
        elif 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        else:
            raise ValueError("No datetime or timestamp column found")

        self.target_column = None
        for possible_name in self.pollutant_mapping[self.target_pollutant]:
            if possible_name in self.data.columns:
                self.target_column = possible_name
                break

        if self.target_column is None:
            raise ValueError(f"Target pollutant {self.target_pollutant.upper()} not found. "
                             f"Available columns: {self.data.columns.tolist()}")

        if self.target_column != self.target_pollutant:
            self.data[self.target_pollutant] = self.data[self.target_column]
            self.data.drop(self.target_column, axis=1, inplace=True)

        self.data[self.target_pollutant] = self.data[self.target_pollutant].fillna(method='ffill').fillna(method='bfill')
        self.data = self.data.sort_values(['station', 'timestamp']).reset_index(drop=True)

        available_stations = self.data['station'].unique()
        if num_stations is not None and num_stations < len(available_stations):
            selected_stations = available_stations[:num_stations]
            self.data = self.data[self.data['station'].isin(selected_stations)]

        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['weekday'] = self.data['timestamp'].dt.dayofweek

        available_columns = self.data.columns.tolist()
        base_features = [self.target_pollutant, 'hour', 'weekday']

        pollutant_features = []
        all_pollutant_names = [name for names_list in self.pollutant_mapping.values() for name in names_list]
        for col in available_columns:
            if col in all_pollutant_names and col != self.target_pollutant and col not in base_features:
                pollutant_features.append(col)

        if self.target_pollutant != 'aqi' and 'aqi' in available_columns and 'aqi' not in pollutant_features:
            pollutant_features.append('aqi')

        # Removed weather_features logic here

        self.feature_cols = base_features + pollutant_features
        self.continuous_cols = [col for col in self.feature_cols if col not in ['hour', 'weekday']]

        self.feature_scaler = None
        self.target_scaler = None

        if self.is_no2_target:
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))

            target_data_original = self.data[self.target_pollutant].values.reshape(-1, 1)
            self.target_scaler.fit(target_data_original)

            continuous_data = self.data[self.continuous_cols].values
            self.data[self.continuous_cols] = self.feature_scaler.fit_transform(continuous_data)

        target_data = self.data[self.target_pollutant].values
        self.target_mean = np.mean(target_data)
        self.target_std = np.std(target_data)
        self.target_min = np.min(target_data)
        self.target_max = np.max(target_data)

        self.stations = self.data['station'].unique()
        self.train_data = []
        self.val_data = []
        self.test_data = []

        for station in self.stations:
            station_data = self.data[self.data['station'] == station].reset_index(drop=True)

            min_length = self.sequence_length + self.target_steps + 50
            if len(station_data) < min_length:
                continue

            train_end = int(len(station_data) * 0.8)
            val_end = int(len(station_data) * 0.9)

            train_station = station_data[:train_end].copy()
            val_station = station_data[train_end:val_end].copy()
            test_station = station_data[val_end:].copy()

            self.train_data.extend(self._create_sequences(train_station, station))
            self.val_data.extend(self._create_sequences(val_station, station))
            self.test_data.extend(self._create_sequences(test_station, station))

        if len(self.train_data) == 0:
            raise ValueError("No training sequences created")

        print(f"Dataset ready: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test")

    def _create_sequences(self, station_data, station_name):
        if len(station_data) < self.sequence_length + self.target_steps:
            return []

        sequences = []
        features = station_data[self.feature_cols].values
        target_values = station_data[self.target_pollutant].values

        for i in range(len(features) - self.sequence_length - self.target_steps + 1):
            input_seq = features[i:i + self.sequence_length]
            target_seq = target_values[i + self.sequence_length:i + self.sequence_length + self.target_steps]

            if not (np.isnan(input_seq).any() or np.isnan(target_seq).any() or
                    np.isinf(input_seq).any() or np.isinf(target_seq).any()):
                sequences.append((input_seq, target_seq, station_name))

        return sequences

    def get_train_data(self):
        if not self.train_data:
            return None, None
        inputs = torch.stack([torch.from_numpy(seq[0]).float() for seq in self.train_data])
        targets = torch.stack([torch.from_numpy(seq[1]).float() for seq in self.train_data])
        return inputs, targets

    def get_val_data(self):
        if not self.val_data:
            return None, None
        inputs = torch.stack([torch.from_numpy(seq[0]).float() for seq in self.val_data])
        targets = torch.stack([torch.from_numpy(seq[1]).float() for seq in self.val_data])
        return inputs, targets

    def get_test_data(self):
        if not self.test_data:
            return None, None
        inputs = torch.stack([torch.from_numpy(seq[0]).float() for seq in self.test_data])
        targets = torch.stack([torch.from_numpy(seq[1]).float() for seq in self.test_data])
        return inputs, targets

    def get_feature_scaler(self):
        return self.feature_scaler

    def get_target_scaler(self):
        return self.target_scaler

    def inverse_transform_target(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        if self.is_no2_target and self.target_scaler is not None:
            original_shape = predictions.shape
            predictions_flat = predictions.reshape(-1, 1)
            predictions_denorm = self.target_scaler.inverse_transform(predictions_flat)
            return predictions_denorm.reshape(original_shape)
        return np.asarray(predictions)

    def get_target_pollutant_index(self):
        try:
            return self.feature_cols.index(self.target_pollutant)
        except ValueError:
            return 0

    def get_target_statistics(self):
        return {
            'mean': self.target_mean,
            'std': self.target_std,
            'min': self.target_min,
            'max': self.target_max,
            'is_normalized': self.is_no2_target
        }

    def get_data_info(self):
        return {
            'target_pollutant': self.target_pollutant,
            'target_column_original': self.target_column,
            'num_stations': len(self.stations),
            'feature_columns': self.feature_cols,
            'continuous_columns': self.continuous_cols,
            'sequence_length': self.sequence_length,
            'target_steps': self.target_steps,
            'train_sequences': len(self.train_data),
            'val_sequences': len(self.val_data),
            'test_sequences': len(self.test_data),
            'normalization_method': 'MinMax' if self.is_no2_target else 'RevIN',
            'is_no2_target': self.is_no2_target,
            'target_statistics': self.get_target_statistics(),
            'target_pollutant_index': self.get_target_pollutant_index(),
            'supports_aqi': True
        }
