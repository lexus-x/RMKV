import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class VisionEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        # We will use torch.hub or transformers for dinov2
        # Use transformers AutoModel
        # No silent fallback — fail loudly if pretrained weights missing.
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.parameters():
            param.requires_grad = False
        self.out_dim = 768

    def forward(self, x):
        # x is (B, num_views, 3, H, W) or (B, 3, H, W)
        if len(x.shape) == 5:
            B, V, C, H, W = x.shape
            x = x.reshape(B * V, C, H, W)
            outputs = self.model(x)
            features = outputs.last_hidden_state  # (B*V, S, 768)
            _, S, D = features.shape
            return features.reshape(B, V * S, D)
        outputs = self.model(x)
        return outputs.last_hidden_state  # (B, S, 768)


class LanguageEncoder(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolLM-135M"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.parameters():
            param.requires_grad = False
        self.out_dim = 576

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask # (B, 576)


class ProprioEncoder(nn.Module):
    """Encodes proprio. If 3-D input (B, obs_len, dim), uses LAST timestep only."""
    def __init__(self, in_dim=15, out_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Mish(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]  # last timestep
        assert x.shape[-1] == self.in_dim, f"proprio dim {x.shape[-1]} != {self.in_dim}"
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim=768, lang_dim=576, d_model=256, n_heads=4, n_layers=2):
        super().__init__()
        self.vis_proj = nn.Linear(vision_dim, d_model)
        self.lang_proj = nn.Linear(lang_dim, d_model)
        
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.Mish(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_layers)
        ])

    def forward(self, vision_feat, lang_feat):
        # vision_feat: (B, seq_v, 768)
        # lang_feat: (B, 576) -> expand to (B, 1, 576)
        
        V = self.vis_proj(vision_feat) # (B, seq_v, 256)
        L = self.lang_proj(lang_feat.unsqueeze(1)) # (B, 1, 256)
        
        # Cross attention: Query=Lang, Key/Value=Vision
        # We want a single condition vector per batch
        x = L
        for i in range(len(self.layers)):
            attn_out, _ = self.layers[i](query=x, key=V, value=V)
            x = self.norm1[i](x + attn_out)
            ffn_out = self.ffn[i](x)
            x = self.norm2[i](x + ffn_out)
            
        return x.squeeze(1) # (B, 256)
