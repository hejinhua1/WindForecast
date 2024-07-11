import torch 
import torch.nn.functional as F
from model import TPALSTM
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import optuna
import mlflow
from tqdm import tqdm
import yaml
from train_scripts.dataset import CustomDataset
# from utils import get_train_data_date

import warnings
warnings.filterwarnings("ignore")


with open("/home/hjh/WindForecast/train_scripts/config.yaml", "r") as f:
    config = yaml.safe_load(f)
# 读取YAML文件
with open("/home/hjh/WindForecast/station_config.yaml", "r") as f:
    station_config = yaml.safe_load(f)

train_station_id = 0

# 提取id到场站的映射
id_station_mapping = {}
for station, attributes in station_config.items():
    id_station_mapping[attributes['id']] = station

pred_step = 16
EPOCH_NUM = 50

# 生成task_name
station_name = id_station_mapping[train_station_id]
task_name = f"guangdong-wind-{station_name}"

print(task_name)

mlflow.set_experiment(task_name)

train_set = CustomDataset(mode="train", id=train_station_id)
val_set = CustomDataset(mode="val", id=train_station_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#超参设置的地方
def suggest_hyperparameters(trial):
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048]) #[64, 148, 456]
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
    hidden_size = trial.suggest_categorical("hidden_size", [42, 48, 56, 64])

    return batch_size, lr, n_layers, hidden_size

def train(model, train_loader, optimizer, epoch):
    model.train()
    
    step_num = epoch * len(train_loader)
    for Xtrain, future_info, station_id, ytrain in tqdm(train_loader, total=len(train_loader)):
        Xtrain = Xtrain.to(device).float()
        future_info = future_info.to(device).float()
        station_id = station_id.to(device) #
        # station_id = None
        
        ytrain = ytrain.to(device).float()

        weights = torch.tensor([.2, .3, .4, .1, .2, .3, .4, .1]).to(device)
        #滚动预测6次,每次预测16个点
        loss_val = 0
        for i in range(int(config["pred_len"]/pred_step)):
            Xtrain = torch.cat([Xtrain, torch.cat([future_info[:, i*pred_step:(i+1)*pred_step].clone(), 
                                                   torch.zeros((Xtrain.shape[0], pred_step, config["label_size"])).to(device)
                                                   ], dim=-1)], dim=1)
            ypred = model(Xtrain, station_id)
            ypred = ypred.reshape(ypred.shape[0], pred_step, config["label_size"])
            loss = F.mse_loss(ypred, ytrain[:, i*pred_step:(i+1)*pred_step], reduction="none")
            weighted_loss = loss * weights  # 应用权重
            final_loss = weighted_loss.mean()  # 计算最终加权损失
            loss_val += final_loss.item()
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            #将模型的输出作为下一次的输入
            #使用teacher forcing, 开始使用真实值,逐渐使用模型的输出
            # if epoch < 4:
            #     use_teacher = True
            # elif epoch < 5:
            #     use_teacher = np.random.rand() < 0.5
            # elif epoch < 10:
            #     use_teacher = np.random.rand() < 0.4
            # else:
            #     use_teacher = False
            use_teacher = False

            if use_teacher:
                future_pred = ytrain[:, i*pred_step:(i+1)*pred_step].clone()
            else:
                future_pred = ypred.detach()

            Xtrain[:, -pred_step:, -config["label_size"]:] = future_pred #填充0
            Xtrain = Xtrain[:, pred_step:]           #滚动
            
        step_num += 1
        
        loss_val /= int(config["pred_len"]/pred_step)
        mlflow.log_metric("train_loss", loss_val, step=step_num)

def test(model, test_loader):
    model.eval()

    labels = []
    preds = []
    with torch.no_grad():
        for Xtrain, future_info, station_id, ytrain in test_loader:
            Xtrain = Xtrain.to(device).float()
            future_info = future_info.to(device).float()
            station_id = station_id.to(device)
            # station_id = None
            
            pred_res = []
            for i in range(int(config["pred_len"]/pred_step)):
                Xtrain = torch.cat([Xtrain, torch.cat([future_info[:, i*pred_step:(i+1)*pred_step].clone(), 
                                                    torch.zeros((Xtrain.shape[0], pred_step, config["label_size"])).to(device)
                                                    ], dim=-1)], dim=1)
                ypred = model(Xtrain, None)
                ypred = ypred.reshape(ypred.shape[0], pred_step, config["label_size"])
                pred_res.append(ypred.cpu().numpy())
                Xtrain[:, -pred_step:, -config["label_size"]:] = ypred.detach() #填充0
                Xtrain = Xtrain[:, pred_step:]           #滚动
            labels.append(ytrain.cpu().numpy())
            preds.append(np.concatenate(pred_res, axis=1))
            
            
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    avg_mae = mean_absolute_error(labels.reshape(labels.shape[0], -1), preds.reshape(labels.shape[0], -1))
    avg_mape = mean_absolute_percentage_error(labels.reshape(labels.shape[0], -1), preds.reshape(labels.shape[0], -1))
    avg_loss = F.mse_loss(torch.tensor(labels), torch.tensor(preds), reduction="none")
    avg_loss = avg_loss * torch.tensor([.2, .3, .4, .1, .2, .3, .4, .1]) #weights
    avg_loss = avg_loss.mean()
    return avg_mae, avg_mape, avg_loss.item()

def objective(trial):
    best_val_loss = float('Inf')
    # if True:
    with mlflow.start_run() as run_ins:
        trial.set_user_attr("run_id", run_ins.info.run_id) #记录试验的run_id

        #获取超参
        batch_size, lr, n_layers, hidden_size = suggest_hyperparameters(trial)

        #注册超参
        mlflow.log_params({"batch_size": batch_size, "lr": lr, "n_layers": n_layers, "hidden_size": hidden_size})

        #实例化模型
        model = TPALSTM(input_size=config["feats_size"], 
                        output_size=config["label_size"],
                        hidden_size=hidden_size, 
                        obs_len=config["obs_len"]+pred_step, #设计到拼接的问题,所以这里要加3 
                        pred_len=pred_step, #滚动预测, 预测3个时刻,滚动8次
                        n_layers=n_layers)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches, eta_min=1e-5)

        #获取数据
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=24)

        patience_time = 0
        for epoch in range(EPOCH_NUM):

            train(model, train_loader, optimizer, epoch)
            # scheduler.step() #是否使用？

            avg_mae, avg_mape, avg_loss = test(model, val_loader)

            if avg_mae <= best_val_loss:
                best_val_loss = avg_mae
                patience_time = 0 #清空计次
              
                mlflow.pytorch.log_model(model, artifact_path="pred_model")
                #直接注册模型
                # mlflow.register_model(f"runs:/{run_ins.info.run_id}/pred_model", f"{task_name}", tags={"date": str(datetime.now())})
            mlflow.log_metric("val_mae", avg_mae, step=epoch)
            mlflow.log_metric("val_mape", avg_mape, step=epoch)
            mlflow.log_metric("val_loss", avg_loss, step=epoch)

            patience_time += 1
            if patience_time > config["patience"]:
                break
    return best_val_loss
    
def main():
    study_name = f"{task_name}"
    try: #先从数据库中加载
        study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")
    except KeyError: #若数据库中不存在，则创建
        try:
            study = optuna.create_study(study_name=study_name, direction="minimize", storage="sqlite:///optuna.db")
        except:
            study = optuna.create_study(study_name=study_name, direction="minimize")

    study.optimize(objective, n_trials=10)


if __name__ == "__main__":
    main()