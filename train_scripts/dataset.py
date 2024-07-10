import torch, os
import pandas as pd
import numpy as np
from joblib import load
from torch.utils.data import Dataset
import yaml

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

class CustomDataset(Dataset):
    def __init__(self, data_dir="/home/hjh/WindForecast/data", mode="train", step=1, id=None) -> None:
        super().__init__()
        self.nwp_scaler = load(os.path.join(data_dir, "scaler_nwp.joblib"))
        self.wind_scaler = load(os.path.join(data_dir, "scaler_wind.joblib"))
        self.power_scaler = load(os.path.join(data_dir, "scaler_power.joblib"))
        
        self.load(data_dir=data_dir, mode=mode, step=step, id=id)
        
    def load(self, data_dir, mode, step, id):
        data = pd.read_feather(os.path.join(data_dir, "data.feather"))
        latest_time = max(data.TIMESTAMP)
        one_week_ago = latest_time - pd.Timedelta(weeks=1)
        two_weeks_ago = latest_time - pd.Timedelta(weeks=2)
        
        if mode == "train":
            data = data[(data.TIMESTAMP < two_weeks_ago)]
        elif mode == "val":
            data = data[(data.TIMESTAMP >= two_weeks_ago) & (data.TIMESTAMP < one_week_ago)]
        else:
            data = data[(data.TIMESTAMP >= one_week_ago)]
        
        columns = ['10', '10_ewm_7', '10_ewm_3', '4','5','6','7', 'predict_duration', #NWP 0-6
                   'wind_record', 'wind_record_medfilt', 'wind_record_ewm_7', 'wind_record_ewm_3',   #wind 7-8
                   'power_unit', 'power_medfilt_unit', 'power_unit_ewm_7', 'power_unit_ewm_3',      #power 9-10
                   'valid']
        
        self.history = []
        self.condition = []
        self.label = []
        self.station_ids = []
        
        for station_id in data.id.unique():
            if id != None and id != station_id:
                continue
            
            data_ = data[data.id == station_id].dropna()
            data_np = data_[columns].values
            
            data_np[:, :7] = self.nwp_scaler.transform(data_np[:, :7])
            data_np[:, 8:12] = self.wind_scaler.transform(data_np[:, 8:12])
            data_np[:, 12:16] = self.power_scaler.transform(data_np[:, 12:16])
            
            window_size = config['obs_len'] + config['pred_len']
            for i in range(0, len(data_np) - window_size, step):
                window_end = i + window_size
                window_data = data_np[i: window_end]
                
                if sum(window_data[:, -1]) < window_size:
                    continue
                self.history.append(window_data[:config['obs_len'], :-1])
                self.condition.append(window_data[config['obs_len']:, :8])
                self.label.append(window_data[config['obs_len']:, 8:16])
                self.station_ids.append(station_id)
                
    def __len__(self):
        return len(self.history)
    
    def __getitem__(self, index):
        return self.history[index], self.condition[index], self.station_ids[index], self.label[index]

if __name__ == "__main__":
    dataset = CustomDataset()
    for batch in dataset:
        print(batch[0].shape)
        print(len(dataset))