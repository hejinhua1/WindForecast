import os
import gradio as gr
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
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
# 示例场站数据和文件路径格式
station_names = ["珍珠湾", "峡沙", "玄武A", "融星", "璧青湾",
                 "峡阳A", "尚洋湾", "盈和A", "盈和B", "慈航A", "慈航B",
                 "彩石滩A", "蕴阳", "勒门", "菩提", "澎湃A",
                 "澎湃B", "玄武B", "峡阳B"]
data_type_mapper = {
    "风速": "wind",
    "功率": "power"
}
# 定义相对路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler_nwp = load(os.path.join(base_path, "data", "scaler_nwp.joblib"))
scaler_wind = load(os.path.join(base_path, "data", "scaler_wind.joblib"))
scaler_power = load(os.path.join(base_path, "data", "scaler_power.joblib"))

with open(os.path.join(base_path, "train_scripts", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

logged_model = os.path.join(base_path, "train_scripts", "mlruns", "927062044067789145", "99d3df273b7f41d889df08bd30d4fe12", "artifacts", "pred_model", "data", "model.pth")
model = torch.load(logged_model, map_location=torch.device('cpu'))
model.to("cpu").eval()

with open(os.path.join(base_path, "station_config.yaml"), "r", encoding="utf-8") as f:
    station_cfg = yaml.safe_load(f)

name_mapper = {station_cfg[key]['code_nwp']: key for key in station_cfg.keys()}
id_mapper = {key: station_cfg[key]['id'] for key in station_cfg.keys()}
name_mappper_reverse = {var: key for key, var in name_mapper.items() if key is not None}


def load_data():
    data = pd.read_feather(os.path.join(base_path, "data", "data.feather"))
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
        for i in range(pred_hours // pred_step):
            input_x_tensor = torch.cat([input_x_tensor,
                                        torch.cat([input_extra_tensor[:, i * pred_step:(i + 1) * pred_step].clone(),
                                                   torch.zeros(
                                                       (input_x_tensor.shape[0], pred_step, config["label_size"]))],
                                                  dim=-1)]
                                       , dim=1).float()
            ypred = model(input_x_tensor, station_id_tensor)
            ypred = ypred.reshape(-1, pred_step, config["label_size"])
            pred_res.append(ypred.cpu().numpy())
            input_x_tensor[:, -pred_step:, -config["label_size"]:] = ypred
            input_x_tensor = input_x_tensor[:, pred_step:]
    return np.concatenate(pred_res, axis=1).reshape(-1, config["label_size"])




def get_pred_real_data(start_time, end_time, pred_station):
    data_nwp, data_wind, data_power = load_data()

    time_areas = []
    for time_ in pd.date_range(start=start_time, end=end_time, freq='1D'):
        time_past = pd.DataFrame(
            {"TIMESTAMP": pd.date_range(end=time_, periods=config["obs_len"], freq='15T') - pd.Timedelta(minutes=15)})
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
            if pred_station is not None and pred_station != station:
                continue
            station_id = vars["id"]
            cap = vars["capacity"]

            # 合成观测数据
            # nwp
            nwp_ = data_nwp[data_nwp.id == station_id]
            nwp_ = pd.merge(time_all, nwp_, on="TIMESTAMP", how="left")
            nwp_ = nwp_.ffill().bfill()

            nwp_['10_ewm_7'] = nwp_['10'].ewm(alpha=0.7).mean()
            nwp_['10_ewm_3'] = nwp_['10'].ewm(alpha=0.3).mean()

            # history_nwp = nwp_[nwp_.TIMESTAMP < start_time]
            history_nwp = pd.merge(time_past, nwp_, on="TIMESTAMP", how="left")
            history_nwp_array = history_nwp[['10', '10_ewm_7', '10_ewm_3', '4', '5', '6', '7', 'predict_duration']].values
            history_nwp_array[:, :7] = scaler_nwp.transform(history_nwp_array[:, :7])

            # wind
            wind_ = data_wind[data_wind.id == station_id]
            wind_past = pd.merge(time_past, wind_, on="TIMESTAMP", how="left")
            wind_past = wind_past.ffill().bfill()

            wind_past['wind_record_medfilt'] = medfilt(wind_past['wind_record'], kernel_size=5)
            wind_past['wind_record_ewm_7'] = wind_past['wind_record'].ewm(alpha=0.7).mean()
            wind_past['wind_record_ewm_3'] = wind_past['wind_record'].ewm(alpha=0.3).mean()

            wind_future = pd.merge(time_future, wind_, on="TIMESTAMP", how="left")

            history_wind = wind_past[
                ['wind_record', 'wind_record_medfilt', 'wind_record_ewm_7', 'wind_record_ewm_3']].values
            history_wind = scaler_wind.transform(history_wind)

            # power
            power_ = data_power[data_power.id == station_id]
            power_past = pd.merge(time_past, power_, on="TIMESTAMP", how="left")
            power_past = power_past.ffill().bfill()

            power_past['power_unit_ewm_7'] = power_past['power_unit'].ewm(alpha=0.7).mean()
            power_past['power_unit_ewm_3'] = power_past['power_unit'].ewm(alpha=0.3).mean()

            power_future = pd.merge(time_future, power_, on="TIMESTAMP", how="left")

            history_power = power_past[['power_unit', 'power_medfilt_unit', 'power_unit_ewm_7', 'power_unit_ewm_3']].values
            history_power = scaler_power.transform(history_power)

            # 合并
            history = np.concatenate([history_nwp_array, history_wind, history_power], axis=1)
            history_tensor = torch.from_numpy(history).unsqueeze(0).float()

            # 未来天气预报数据
            # future_nwp = nwp_[nwp_.TIMESTAMP >= start_time]
            future_nwp = pd.merge(time_future, nwp_, on="TIMESTAMP", how="left")
            future_nwp_array = future_nwp[['10', '10_ewm_7', '10_ewm_3', '4', '5', '6', '7', 'predict_duration']].values
            future_nwp_array[:, :7] = scaler_nwp.transform(future_nwp_array[:, :7])
            future_tensor = torch.from_numpy(future_nwp_array).unsqueeze(0).float()

            # 场站数据
            id_tensor = torch.tensor([station_id]).long()
            pred_times = 1
            pred_step = 16
            preds = infer(model, history_tensor, future_tensor, id_tensor, pred_step, config["pred_len"] * pred_times)

            # preds_label = scaler_wind.inverse_transform(preds[:, -1:]).reshape(-1)
            preds_wind = scaler_wind.inverse_transform(preds[:, :config["label_size"] // 2])
            preds_wind = np.clip(preds_wind, 0, None)
            preds_power = scaler_power.inverse_transform(preds[:, config["label_size"] // 2:])
            preds_power = np.clip(preds_power, 0, 1)
            preds_power *= cap

            df_wind = pd.DataFrame(preds_wind,
                                   columns=['wind_record_pred', 'wind_record_medfilt_pred', 'wind_record_ewm_7_pred',
                                            'wind_record_ewm_3_pred'])
            df_wind = pd.concat([time_future, df_wind, wind_future['wind_record'], future_nwp['10']], axis=1)

            df_power = pd.DataFrame(preds_power,
                                    columns=['power_pred', 'power_medfilt_pred', 'power_ewm_7_pred', 'power_ewm_3_pred'])
            if 'power' not in power_future:
                power_future['power'] = power_future['power_unit'] * cap
            df_power = pd.concat([time_future, df_power, power_future['power']], axis=1)

            # df_wind.to_csv(f"/home/hjh/WindForecast/outputs/infer/{station}_wind_{pred_start_str}_{pred_end_str}.csv",
            #                index=False)
            # df_power.to_csv(f"/home/hjh/WindForecast/outputs/infer/{station}_power_{pred_start_str}_{pred_end_str}.csv",
            #                 index=False)
    return df_wind, df_power



def add_one_day(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    new_date_obj = date_obj + timedelta(days=1)
    new_date_str = new_date_obj.strftime('%Y%m%d')
    return new_date_str



def visualize_data(date, data_type, pred_station):
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date, "%Y%m%d")

    # 创建 start_time 和 end_time
    start_time = date.replace(hour=6, minute=0, second=0)
    end_time = start_time + timedelta(days=1)

    # 将 datetime 对象转换为所需的字符串格式
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    wind_df, power_df = get_pred_real_data(start_time, end_time, pred_station)
    if data_type == "风速":
        fig = px.line(wind_df, x='TIMESTAMP', y=['wind_record_pred', '10'], title=f"{pred_station} 风速")
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="风速",
            legend_title_text="预测 vs 观测"
        )
        fig.for_each_trace(lambda trace: trace.update(name={'wind_record_pred': '预测风速', '10': '实测风速'}[trace.name]))
    elif data_type == "功率":
        fig = px.line(power_df, x='TIMESTAMP', y=['power_pred', 'power'], title=f"{pred_station} 功率")
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="功率",
            legend_title_text="预测 vs 观测"
        )
        fig.for_each_trace(lambda trace: trace.update(name={'power_pred': '预测功率', 'power': '实测功率'}[trace.name]))
    else:
        return "Invalid data type selected."

    return fig


def predict_from_csv(csv_file, data_type, station):
    #TODO: 读取上传的CSV文件
    # 读取上传的CSV文件
    df = pd.read_csv(csv_file.name)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()

    wind_df, power_df = get_pred_real_data(start_time, end_time, station)

    if data_type == "风速":
        fig = px.line(wind_df, x='TIMESTAMP', y=['wind_record_pred', '10'], title=f"{station} 风速")
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="风速",
            legend_title_text="预测 vs 观测"
        )
        fig.for_each_trace(lambda trace: trace.update(name={'wind_record_pred': '预测风速', '10': '实测风速'}[trace.name]))
    elif data_type == "功率":
        fig = px.line(power_df, x='TIMESTAMP', y=['power_pred', 'power'], title=f"{station} 功率")
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="功率",
            legend_title_text="预测 vs 观测"
        )
        fig.for_each_trace(lambda trace: trace.update(name={'power_pred': '预测功率', 'power': '实测功率'}[trace.name]))
    else:
        return "Invalid data type selected."

    return fig


with gr.Blocks() as demo:
    gr.Markdown("# 风速和功率数据可视化")
    gr.Markdown(" 选择场站和数据类型进行可视化")

    with gr.Tab("从日期选择"):
        with gr.Row():
            date = gr.Dropdown(choices=["20230717", "20230718", "20230719"], label="选择日期")
            data_type = gr.Dropdown(choices=["风速", "功率"], label="选择数据类型")
            station = gr.Dropdown(choices=list(station_names), label="选择场站")
        output = gr.Plot()
        plot = gr.Button("可视化")
        plot.click(fn=visualize_data, inputs=[date, data_type, station], outputs=output)

    with gr.Tab("从CSV文件"):
        csv_file = gr.File(label="上传CSV文件")
        with gr.Row():
            data_type = gr.Dropdown(choices=["风速", "功率"], label="选择数据类型")
            station = gr.Dropdown(choices=list(station_names), label="选择场站")
        output_csv = gr.Plot()
        plot_csv = gr.Button("预测")
        plot_csv.click(fn=predict_from_csv, inputs=[csv_file, data_type, station], outputs=output_csv)

demo.launch()