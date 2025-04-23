import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, board_size=(8, 8), n_card_types=10, hidden_dim=128):
        super().__init__()
        self.board_size = board_size
        self.n_card_types = n_card_types
        
        # 卷积层处理输入
        self.conv1 = nn.Conv2d(n_card_types, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 计算展平后的特征维度
        self.flat_dim = 64 * board_size[0] * board_size[1]
        
        # 全连接层
        self.fc1 = nn.Linear(self.flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, board_size[0] * board_size[1])
        
        # Dropout 层防止过拟合
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平
        x = x.view(-1, self.flat_dim)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 输出动作概率
        x = self.fc_out(x)
        return F.softmax(x, dim=1)
        
    def get_action(self, state, valid_actions=None, deterministic=False):
        """获取动作"""
        with torch.no_grad():
            action_probs = self(state)
            
            # 如果提供了有效动作，将无效动作的概率设为0
            if valid_actions is not None:
                mask = torch.zeros_like(action_probs)
                mask[:, valid_actions] = 1
                action_probs = action_probs * mask
                action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1)
            else:
                action = torch.multinomial(action_probs, 1)
            
            return action.item() 