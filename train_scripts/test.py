import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
import torch
import yaml
import pandas as pd
import numpy as np
from joblib import load
from torch.utils.data import DataLoader
from train_scripts.dataset import CustomDataset

import warnings
warnings.filterwarnings("ignore")

data_storage_dir = "data_storage"
model_name = "wind-speed-pred"
model_version = "186"
pred_station_id = 0
pred_station_id = 4


scaler_nwp = load("/home/hjh/WindForecast/data/scaler_nwp.joblib")
scaler_wind = load("/home/hjh/WindForecast/data/scaler_wind.joblib")
scaler_power = load("/home/hjh/WindForecast/data/scaler_power.joblib")

# with open("station_config.yaml", "r") as f:
#     station_conf = yaml.safe_load(f)
with open("/home/hjh/WindForecast/train_scripts/config.yaml", "r") as f:
    config = yaml.safe_load(f)
pred_step = 16

model_id = "23418224017a4fd1b275916e2b125172" #2
model_id = "d19a7ba0194e4450b156ee7268e6e708" #3
model_id = "5e9d8fb2188641e8bf33e975f69541c7" #4
model_id = "4b9f51a2ccec49b3938cffb076932d95" #5
model_id = "7094cc3657674942964893c62ffa4aa9" #6

logged_model = '/home/hjh/WindForecast/train_scripts/mlruns/927062044067789145/99d3df273b7f41d889df08bd30d4fe12/artifacts/pred_model/data/model.pth'
logged_model = '/home/hjh/WindForecast/train_scripts/mlruns/120797431678840380/0f7d15a9b3384dd6b3cd5474cd65c8c5/artifacts/pred_model/data/model.pth'
model = torch.load(logged_model)
model.to("cpu").eval()
    

def infer(model, 
          input_x_tensor, 
          input_extra_tensor, 
          station_id_tensor, 
          pred_step=6, 
          pred_hours=36):
    
    with torch.no_grad():
        pred_res = []
        for i in range(pred_hours//pred_step):
            input_x_tensor = torch.cat([input_x_tensor, 
                                        torch.cat([input_extra_tensor[:, i*pred_step:(i+1)*pred_step].clone(), 
                                                    torch.zeros((input_x_tensor.shape[0], pred_step, config["label_size"]))], dim=-1)]
                                                    , dim=1).float()            
            ypred = model(input_x_tensor, station_id_tensor)
            ypred = ypred.reshape(-1, pred_step, config["label_size"])
            pred_res.append(ypred.cpu().numpy())
            input_x_tensor[:, -pred_step:, -config["label_size"]:] = ypred
            input_x_tensor = input_x_tensor[:, pred_step:]
    return np.concatenate(pred_res, axis=1).reshape(-1, config["label_size"])


with open("/home/hjh/WindForecast/station_config.yaml", "r") as f:
    station_cfg = yaml.safe_load(f)
    
for station, vars in station_cfg.items():
    # stations = []
    labels = []
    nwp_labels = []
    preds = []
    station_id = vars["id"]
    if station_id != pred_station_id:
        continue
    cap = vars["capacity"]
    
    data_ = CustomDataset(mode="test", step=96, id=station_id)
    test_set_loader = DataLoader(data_, batch_size=1, shuffle=False)

    for idx, batch in enumerate(test_set_loader):
        X, future_info, id_, y = batch
        # id_ = None
        pred_res = infer(model, X, future_info, id_, pred_step, int(config["pred_len"]))
        labels.append(y.numpy().reshape(-1, config["label_size"]))
        preds.append(pred_res)

    if len(labels) == 0:
        continue
    labels = np.concatenate(labels)
    # labels_label = scaler_wind.inverse_transform(labels[:, -1:]).reshape(-1)
    labels_wind = scaler_wind.inverse_transform(labels[:, :config["label_size"]//2])
    labels_power = scaler_power.inverse_transform(labels[:, config["label_size"]//2:])
    labels_power *= cap

    preds = np.concatenate(preds)
    # preds_label = scaler_wind.inverse_transform(preds[:, -1:]).reshape(-1)
    preds_wind = scaler_wind.inverse_transform(preds[:, :config["label_size"]//2])
    preds_power = scaler_power.inverse_transform(preds[:, config["label_size"]//2:])
    preds_power *= cap

    # nwp_labels = np.concatenate(nwp_labels)
    # nwp_labels = nwp_labels * scaler_nwp.scale_[-12] + scaler_nwp.mean_[-12]

    data_dict = {
        'wind_record_label': labels_wind[:, 0],
        'wind_record_medfilt_label': labels_wind[:, 1],
        'wind_record_ewm_7_label': labels_wind[:, 2], 
        'wind_record_ewm_3_label': labels_wind[:, 3],
        'power_label': labels_power[:, 0],
        'power_medfilt_unit_label': labels_power[:, 1], 
        'power_ewm_7_label': labels_power[:, 2], 
        'power_ewm_3_label': labels_power[:, 3],
        'wind_record_pred': preds_wind[:, 0],
        'wind_record_medfilt_pred': preds_wind[:, 1], 
        'wind_record_ewm_7_pred': preds_wind[:, 2], 
        'wind_record_ewm_3_pred': preds_wind[:, 3],
        'power_pred': preds_power[:, 0],
        'power_medfilt_unit_pred': preds_power[:, 1], 
        'power_ewm_7_pred': preds_power[:, 2], 
        'power_ewm_3_pred': preds_power[:, 3], 
    }

    df = pd.DataFrame(data_dict)
    power_mape = np.mean(np.abs(df['power_label']-df['power_pred']) / cap)
    df.to_csv("/home/hjh/WindForecast/outputs/test/{}_{:.4f}.csv".format(station, power_mape))