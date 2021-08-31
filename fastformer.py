import torch 
from torch import nn

class FastformerBlock(nn.Module):
    
    def __init__(self, d_model, n_head=1):
        
        super().__init__()
        
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
                
        self.project_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.q_proj = nn.Linear(d_model // n_head, 1)
        self.k_proj = nn.Linear(d_model // n_head, 1)
        self.v_trans = nn.Linear(d_model, d_model)
        
    def get_qkv(self, x):
        qkv = self.project_qkv(x)
        return torch.chunk(qkv, 3, dim=-1)
    
    def forward(self, x):
        
        B, T, C = x.size()
        
        q, k, v = self.get_qkv(x)
        
        # Query attention
        q_heads = q.view(B, T, self.n_head, self.d_model // self.n_head)
        alpha = self.q_proj(q_heads) # B, T, H, 1 / np.sqrt(self.d_model) 
        alpha = torch.softmax(alpha, dim=1)
        q_att = (alpha * q_heads).sum(1, keepdim=True)
        q_att = q_att.view(B, 1, -1)
        
        # Query-Key interaction
        p = q_att * k # [B, T, d_model]
        p = p.view(B, T, self.n_head, self.d_model // self.n_head) # [B, T, d_model]
        beta = self.k_proj(p) #/ np.sqrt(self.d_model)
        beta = torch.softmax(beta, dim=1)
        k_att = (beta * p).sum(1, keepdim=True)
        k_att = k_att.view(B, 1, -1)
        
        # Key-Value interaction
        u = k_att * v
        r = self.v_trans(u)
        
        return r + q
