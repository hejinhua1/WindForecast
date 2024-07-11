import gradio as gr
import pandas as pd
import joblib
print(gr.__version__)
# 假设你有两个训练好的模型
# wind_speed_model = joblib.load('wind_speed_model.pkl')
# power_model = joblib.load('power_model.pkl')


# def predict(file, wind_farm):
#     # 读取CSV文件
#     data = pd.read_csv(file.name)
#
#     # 选择特定的风电场数据
#     selected_data = data[data['WindFarm'] == wind_farm]
#
#     # 假设你的数据有这些特征用于预测
#     features = selected_data[['feature1', 'feature2', 'feature3']]
#
#     # 进行预测
#     predicted_wind_speed = wind_speed_model.predict(features)
#     predicted_power = power_model.predict(features)
#
#     # 添加预测结果到数据中
#     selected_data['PredictedWindSpeed'] = predicted_wind_speed
#     selected_data['PredictedPower'] = predicted_power
#
#     return selected_data
#
#
# # 创建Gradio界面
# input_file = gr.inputs.File(label="上传CSV数据文件")
# wind_farm = gr.inputs.Dropdown(choices=['WindFarm1', 'WindFarm2', 'WindFarm3'], label="选择风电场")
# output_df = gr.outputs.Dataframe(label="预测结果")
#
# demo = gr.Interface(fn=predict,
#              inputs=[input_file, wind_farm],
#              outputs=output_df,
#              title="海上风电场风速和功率预测",
#              description="上传CSV数据文件并选择风电场，点击预测按钮来获取预测风速和功率。")
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
demo.launch(share=True)
