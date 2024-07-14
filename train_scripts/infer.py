import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
import torch
import yaml
import pandas as pd
import numpy as np
from joblib import load
from glob import glob
from tqdm import tqdm
from scipy.signal import medfilt

import warnings
warnings.filterwarnings("ignore")

data_storage_dir = "data_storage"
model_name = "wind-speed-pred"
model_version = "186"
pred_station_id = 4

scaler_nwp = load("/home/hjh/WindForecast/data/scaler_nwp.joblib")
scaler_wind = load("/home/hjh/WindForecast/data/scaler_wind.joblib")
scaler_power = load("/home/hjh/WindForecast/data/scaler_power.joblib")

# with open("station_config.yaml", "r") as f:
#     station_conf = yaml.safe_load(f)
with open("/home/hjh/WindForecast/train_scripts/config.yaml", "r") as f:
    config = yaml.safe_load(f)
pred_step = 16

# id_station_mapper = {station_conf[key]['id']:key for key in station_conf.keys()}
model_id = "23418224017a4fd1b275916e2b125172" #2
model_id = "d19a7ba0194e4450b156ee7268e6e708" #3
model_id = "5e9d8fb2188641e8bf33e975f69541c7" #4
model_id = "7094cc3657674942964893c62ffa4aa9" #6

logged_model = '/home/hjh/WindForecast/train_scripts/mlruns/927062044067789145/99d3df273b7f41d889df08bd30d4fe12/artifacts/pred_model/data/model.pth'
logged_model = '/home/hjh/WindForecast/train_scripts/mlruns/120797431678840380/0f7d15a9b3384dd6b3cd5474cd65c8c5/artifacts/pred_model/data/model.pth'
model = torch.load(logged_model)
model.to("cpu").eval()

with open("/home/hjh/WindForecast/station_config.yaml", "r") as f:
    station_cfg = yaml.safe_load(f)

name_mapper = {station_cfg[key]['code_nwp']:key for key in station_cfg.keys()}
id_mapper = {key:station_cfg[key]['id'] for key in station_cfg.keys()}
name_mappper_reverse = {var: key for key, var in name_mapper.items() if key is not None}

def load_data(start_time, mode="typhoon"):
    if mode == "infer":
        data_nwp = pd.DataFrame()
        dates = pd.date_range(end=start_time, periods=3, freq='1D')
        for date_ in tqdm(dates, total=len(dates)):
            datetime = date_.strftime("%Y%m%d")

            search_path = "{}/nwp/tqyb{}/*.txt".format("data_infer", datetime)
            files = glob(search_path) #处理一天的数据
            for file_ in files:
                station_name = file_.split('-')[-3]
                if station_name not in name_mapper:
                    continue
                data = pd.read_csv(file_, sep=',', header=None, )
                data['TIMESTAMP'] = pd.to_datetime(data[0])+pd.to_timedelta(data[1])
                pivot_df = data.pivot(index='TIMESTAMP', columns=2, values=3)
                pivot_df.reset_index(inplace=True)
                df_static = data[['TIMESTAMP', 4, 5, 6, 7]].drop_duplicates().set_index('TIMESTAMP')
                result_df = pivot_df.merge(df_static, on='TIMESTAMP')
                result_df = result_df[result_df['TIMESTAMP'].dt.minute.isin([0,15,30,45])] #仅仅保留[0,15,30,45]分钟的数据
                result_df['start_predict_time'] = result_df['TIMESTAMP'][0]
                # result_df['station_name'] = name_mapper[station_name]
                result_df['id'] = id_mapper[name_mapper[station_name]]
                result_df['predict_duration'] = result_df['TIMESTAMP'] - result_df['start_predict_time']
                result_df['predict_duration'] = result_df['predict_duration'].dt.total_seconds() / 60 / 60 / 24 #预测时间差转化为float，
                
                data_nwp = pd.concat([data_nwp, result_df])
        data_nwp_new = pd.DataFrame()
        for id_ in data_nwp.id.unique():
            data_station = data_nwp[data_nwp.id == id_]
            data_station = data_station.sort_values('predict_duration').drop_duplicates(subset='TIMESTAMP', keep='first')
            data_station.rename(columns=lambda x: str(x), inplace=True)
            data_nwp_new = pd.concat([data_nwp_new, data_station])
        
        data_wind = pd.DataFrame()
        rename_cols = {"时间": "TIMESTAMP", 
                    "实时风速-10": "wind_record", 
                    "实时风速-30": "30m", 
                    "实时风速-50": "50m",
                    "实时风速-70": "70m",
                    "实时风速-80": "80m",
                    }
        
        for file_ in glob("data_infer/wind/*.csv"):
            station_name = os.path.basename(file_)[:-4]
            if station_name not in name_mappper_reverse:
                continue
            
            data_ = pd.read_csv(file_, sep=',', header=0, na_values='-', encoding='gbk')
            data_.rename(columns=rename_cols, inplace=True)
            data_["TIMESTAMP"] = pd.to_datetime(data_["TIMESTAMP"])
            # data_["station_name"] = station_name
            data_["id"] = id_mapper[station_name]
            data_ = data_[["TIMESTAMP", "id", "wind_record", "30m", "50m", "70m", "80m"]] 
            data_wind = pd.concat([data_wind, data_])
            
        data_['wind_record'] = data_['wind_record'].astype(float)
        data_['30m'] = data_['30m'].astype(float)
        data_['50m'] = data_['50m'].astype(float)
        data_['70m'] = data_['70m'].astype(float)
        data_['80m'] = data_['80m'].astype(float)
            
        data_power = pd.DataFrame()
        rename_cols = {"时间": "TIMESTAMP", 
                    "实测功率": "power", }
        for file_ in glob("data_infer/power/*.csv"):
            station_name = os.path.basename(file_)[:-4]
            if station_name not in name_mappper_reverse:
                continue

            data_ = pd.read_csv(file_, sep=',', header=0, na_values='-', encoding='gbk')
            data_.rename(columns=rename_cols, inplace=True)
            data_["TIMESTAMP"] = pd.to_datetime(data_["TIMESTAMP"])
            data_["power"] = data_["power"].astype(float)
            cap = station_cfg[station_name]["capacity"]
            data_["power_medfilt"] = medfilt(data_["power"], kernel_size=5)
            data_["power_unit"] = data_["power"] / cap
            data_["power_medfilt_unit"] = data_["power_medfilt"] / cap
            # data_["station_name"] = station_name
            data_["id"] = id_mapper[station_name]
            data_ = data_[["TIMESTAMP", "id", "power", "power_unit", "power_medfilt_unit"]]
            data_power = pd.concat([data_power, data_])
    elif mode == "typhoon": #直接从data.feather 中读取
        data = pd.read_feather(os.path.join("/home/hjh/WindForecast/data", "data.feather"))
        data_nwp_new = data
        data_wind = data
        data_power = data
    return data_nwp_new, data_wind, data_power


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

if True:
    mode = "typhoon" #"infer, typhoon"
    start_time = "2023-7-17 06:00:00" #
    end_time = "2023-7-19 06:00:00"
else:
    mode = "infer"
    start_time = "2024-07-4 06:00:00" #
    end_time = None

if mode == "infer":
    pred_times = 7
else:
    pred_times = 1

data_nwp, data_wind, data_power = load_data(start_time=start_time, mode=mode)

time_areas = []
if mode == "infer":
    # 向前128个时间点
    time_past = pd.DataFrame({"TIMESTAMP": pd.date_range(end=start_time, periods=config["obs_len"], freq='15T') - pd.Timedelta(minutes=15)})
    time_future = pd.DataFrame({"TIMESTAMP": pd.date_range(start=start_time, periods=config["pred_len"]*7, freq='15T')})
    time_areas.append([time_past, time_future])
elif mode == "typhoon":
    for time_ in pd.date_range(start=start_time, end=end_time, freq='1D'):
        time_past = pd.DataFrame({"TIMESTAMP": pd.date_range(end=time_, periods=config["obs_len"], freq='15T') - pd.Timedelta(minutes=15)})
        time_future = pd.DataFrame({"TIMESTAMP": pd.date_range(start=time_, periods=config["pred_len"], freq='15T')})
        time_areas.append([time_past, time_future])

for time_past, time_future in time_areas:
    time_all = pd.concat([time_past, time_future])
    pred_start_str = time_future["TIMESTAMP"].iloc[0].strftime("%Y%m%d%H%M")
    pred_end_str = time_future["TIMESTAMP"].iloc[-1].strftime("%Y%m%d%H%M")

    for station, vars in station_cfg.items():
        # stations = []
        labels = []
        nwp_labels = []
        preds = []
        station_id = vars["id"]
        if station_id != pred_station_id:
            continue
        cap = vars["capacity"]
        
        #合成观测数据
        #nwp
        nwp_ = data_nwp[data_nwp.id == station_id]
        nwp_ = pd.merge(time_all, nwp_, on="TIMESTAMP", how="left")    
        nwp_ = nwp_.ffill().bfill()
        
        nwp_['10_ewm_7'] = nwp_['10'].ewm(alpha=0.7).mean()
        nwp_['10_ewm_3'] = nwp_['10'].ewm(alpha=0.3).mean()
        
        # history_nwp = nwp_[nwp_.TIMESTAMP < start_time]
        history_nwp = pd.merge(time_past, nwp_, on="TIMESTAMP", how="left")
        history_nwp_array = history_nwp[['10', '10_ewm_7', '10_ewm_3', '4','5','6','7', 'predict_duration']].values
        history_nwp_array[:, :7] = scaler_nwp.transform(history_nwp_array[:, :7])

        #wind
        wind_ = data_wind[data_wind.id == station_id]
        wind_past = pd.merge(time_past, wind_, on="TIMESTAMP", how="left")
        wind_past = wind_past.ffill().bfill()
        
        wind_past['wind_record_medfilt'] = medfilt(wind_past['wind_record'], kernel_size=5)
        wind_past['wind_record_ewm_7'] = wind_past['wind_record'].ewm(alpha=0.7).mean()
        wind_past['wind_record_ewm_3'] = wind_past['wind_record'].ewm(alpha=0.3).mean()
        
        wind_future = pd.merge(time_future, wind_, on="TIMESTAMP", how="left")
        
        history_wind = wind_past[['wind_record', 'wind_record_medfilt', 'wind_record_ewm_7', 'wind_record_ewm_3']].values
        history_wind = scaler_wind.transform(history_wind)
        
        #power
        power_ = data_power[data_power.id == station_id]
        power_past = pd.merge(time_past, power_, on="TIMESTAMP", how="left")
        power_past = power_past.ffill().bfill()

        power_past['power_unit_ewm_7'] = power_past['power_unit'].ewm(alpha=0.7).mean()
        power_past['power_unit_ewm_3'] = power_past['power_unit'].ewm(alpha=0.3).mean()
        
        power_future = pd.merge(time_future, power_, on="TIMESTAMP", how="left")

        history_power = power_past[['power_unit', 'power_medfilt_unit', 'power_unit_ewm_7', 'power_unit_ewm_3']].values
        history_power = scaler_power.transform(history_power)
    
        #合并
        history = np.concatenate([history_nwp_array, history_wind, history_power], axis=1)
        history_tensor = torch.from_numpy(history).unsqueeze(0).float()
        
        # 未来天气预报数据
        # future_nwp = nwp_[nwp_.TIMESTAMP >= start_time]
        future_nwp = pd.merge(time_future, nwp_, on="TIMESTAMP", how="left")
        future_nwp_array = future_nwp[['10', '10_ewm_7', '10_ewm_3', '4','5','6','7', 'predict_duration']].values
        future_nwp_array[:, :7] = scaler_nwp.transform(future_nwp_array[:, :7])
        future_tensor = torch.from_numpy(future_nwp_array).unsqueeze(0).float()
        
        #场站数据
        id_tensor = torch.tensor([station_id]).long()
        
        preds = infer(model, history_tensor, future_tensor, id_tensor, pred_step, config["pred_len"]*pred_times)

        # preds_label = scaler_wind.inverse_transform(preds[:, -1:]).reshape(-1)
        preds_wind = scaler_wind.inverse_transform(preds[:, :config["label_size"]//2])
        preds_wind = np.clip(preds_wind, 0, None)
        preds_power = scaler_power.inverse_transform(preds[:, config["label_size"]//2:])
        preds_power = np.clip(preds_power, 0, 1)
        preds_power *= cap

        df_wind = pd.DataFrame(preds_wind, columns=['wind_record_pred', 'wind_record_medfilt_pred', 'wind_record_ewm_7_pred', 'wind_record_ewm_3_pred'])
        df_wind = pd.concat([time_future, df_wind, wind_future['wind_record'], future_nwp['10']], axis=1)

        df_power = pd.DataFrame(preds_power, columns=['power_pred', 'power_medfilt_pred', 'power_ewm_7_pred', 'power_ewm_3_pred'])
        if 'power' not in power_future:
            power_future['power'] = power_future['power_unit'] * cap
        df_power = pd.concat([time_future, df_power, power_future['power']], axis=1)
        
        
        df_wind.to_csv(f"/home/hjh/WindForecast/outputs/infer/{station}_wind_{pred_start_str}_{pred_end_str}.csv", index=False)
        df_power.to_csv(f"/home/hjh/WindForecast/outputs/infer/{station}_power_{pred_start_str}_{pred_end_str}.csv", index=False)