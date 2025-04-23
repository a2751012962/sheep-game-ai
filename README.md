# 兔了个兔 AI 助手

这是一个基于深度学习的 AI 助手，用于为"兔了个兔"游戏提供智能提示。

## 功能特点

- 使用卷积神经网络分析游戏状态
- 预测最优的下一步操作
- RESTful API 接口
- 支持跨域请求

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务器

```bash
python server.py
```

服务器将在 http://localhost:5000 上运行。

## API 接口

### POST /predict

接收当前游戏状态，返回推荐的下一步操作。

请求示例：
```json
{
  "gameState": {
    "nodes": [
      {
        "id": "1",
        "type": "rabbit",
        "x": 100,
        "y": 200,
        "z": 1,
        "canClick": true,
        "isRemoved": false
      }
    ]
  }
}
```

响应示例：
```json
{
  "action": {
    "type": "select",
    "data": {
      "node": {
        "id": "1",
        "type": "rabbit"
      }
    }
  }
}
```

## 部署

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/sheep-game-ai.git
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行服务器：
```bash
python server.py
```

## 技术栈

- Python 3.9+
- PyTorch
- Flask
- Flask-CORS 