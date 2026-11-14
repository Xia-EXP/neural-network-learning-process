# 神经网络学习台

这是一个离线课堂演示项目：浏览器负责交互展示，Python 本地服务负责摄像头采集、视觉特征提取、数据质量检查和分类网络训练。项目使用冻结的 MobileNetV2 ONNX 特征，加上颜色分布特征，训练一个小型分类网络，让学生观察样本、特征、神经元和损失曲线之间的关系。

## 项目特点

- Windows 10/11 64 位可运行；开发模式不需要 API Key 或云端服务。
- 摄像头图像和训练数据默认只保存在本机。
- 支持红、黄、蓝三类样本，采集、训练、暂停、回放、独立检验和课堂配置导出。
- MobileNetV2 作为冻结的视觉特征提取器；每次课堂重新训练的是 1304→32→3 的小型分类网络。
- `web/` 包含课堂页面和可视化资源；`server.py` 提供本机 HTTP 服务。

## 运行开发版

需要 Python 3.12 或更高版本，并安装依赖：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py
```

然后打开服务输出的本机地址，通常是 `http://127.0.0.1:8765/`。如果端口被占用，程序会尝试 8766–8795。

如需给同一课堂局域网内的 GM92 或平板访问，使用 `python server.py --lan`。程序会为本次启动生成访问令牌，并把带令牌的地址打印到启动窗口；令牌也可以通过环境变量 `LEARNING_DESK_LAN_TOKEN` 预先指定。不要把带令牌的地址发布到公共网络。

也可以双击 `一键启动.cmd`，但该脚本适合已经准备好本地 Python 环境的开发目录。原始免安装包中的 Python runtime、依赖库和课堂数据未提交到仓库；请使用虚拟环境安装 `requirements.txt`。

## 模型与许可

`models/mobilenetv2-7.onnx` 来自 [ONNX Model Zoo](https://github.com/onnx/models)，其许可和来源说明见 `models/LICENSE-onnx-models.txt` 与 `models/来源说明.md`。模型 SHA-256 为 `c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f`。

## 数据隐私

课堂照片、训练历史和当前模型属于本地运行数据。仓库没有提交原始课堂采集数据；程序运行后生成的 `data/session.json`、照片和导出文件也应保持在本地，不要提交到公开仓库。

## 测试

开发依赖安装后运行：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

