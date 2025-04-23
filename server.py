from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import sys

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from policy import PolicyNetwork
except ImportError as e:
    print(f"Error importing PolicyNetwork: {str(e)}")
    class PolicyNetwork:
        def __init__(self, *args, **kwargs):
            pass
        def eval(self):
            pass

app = Flask(__name__)
CORS(app)

# 加载模型
try:
    model = PolicyNetwork(board_size=(8, 8), n_card_types=10, hidden_dim=128)
    checkpoint_path = os.path.join(current_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        print("Successfully loaded model checkpoint")
    else:
        print("Warning: checkpoint.pt not found")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "AI Helper Service is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
            
        data = request.get_json()
        game_state = data.get('gameState', {})
        nodes = game_state.get('nodes', [])
        
        # 找到第一个可点击的节点
        for node in nodes:
            if node.get('canClick', False) and not node.get('isRemoved', False):
                return jsonify({
                    'action': {
                        'type': 'select',
                        'data': {
                            'node': node
                        }
                    }
                })
        
        return jsonify({
            'error': 'No clickable nodes available'
        }), 404
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # 添加错误日志
        return jsonify({
            'error': str(e)
        }), 500

def process_game_state(game_state):
    """将游戏状态转换为模型输入格式"""
    nodes = game_state.get('nodes', [])
    print("Processing nodes:", len(nodes))  # 添加调试日志
    
    # 创建一个8x8的空棋盘
    board = np.zeros((10, 8, 8))
    
    for node in nodes:
        if not node['isRemoved'] and node['canClick']:
            # 将节点位置映射到8x8网格
            x = min(int(node['x'] / 100), 7)
            y = min(int(node['y'] / 100), 7)
            # 将节点类型映射到通道
            type_idx = hash(node['type']) % 10
            board[type_idx, y, x] = 1
    
    return torch.FloatTensor(board).unsqueeze(0)

def generate_action(action_index, game_state):
    """根据模型预测生成具体动作"""
    nodes = game_state.get('nodes', [])
    clickable_nodes = [node for node in nodes if not node['isRemoved'] and node['canClick']]
    print("Clickable nodes:", len(clickable_nodes))  # 添加调试日志
    
    if not clickable_nodes:
        return {'action': {'type': 'shuffle'}}
    
    # 选择可点击节点中的第一个
    selected_node = clickable_nodes[0]
    
    return {
        'action': {
            'type': 'select',
            'data': {'node': selected_node}
        }
    }

if __name__ == '__main__':
    # 使用环境变量中的端口，如果没有则默认使用 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 