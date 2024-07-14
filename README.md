
使用 mlflow 来管理试验记录，使用optuna进行超参优化。

在train_scripts目录下 运行 mlflow ui，可以在可视化界面中查看记录。

服务器上运行时，需要进行端口映射，例如需要在本地终端输入：

```ssh -L 8000:localhost:5000 username@remote_host```

然后在浏览器中输入 http://localhost:8000/ 即可查看记录。