import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_pca import PCA
import train_config
config = train_config.config
class FFN(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=450, out_dim=360):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, F_roi):
        # F_roi: (B, 400, 1632)
        return self.net(F_roi)  # (B, 400, out_dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_tokens=1000):
        super().__init__()
        self.pos_embed=nn.Parameter(
            torch.randn(1,max_tokens,dim)
        )

def extract_nodes(Q, kernel_size=3, stride=2):
    batch_size, channels, d,h,w=Q.shape
    blocks=Q.unfold(2,kernel_size,stride).unfold(3, kernel_size,stride).unfold(4,kernel_size,stride)
    node_features=blocks.sum(dim=(-3,-2,-1))
    node_features= node_features.reshape(batch_size, channels, -1)
    node_features= node_features.permute(0,2,1)
    with torch.no_grad():
        mask=node_features.abs().sum(dim=(0,2)) > 0
        valid_indices=torch.where(mask)
    valid_indices=valid_indices[0]
    final_nodes=node_features[:,valid_indices,:]
    # print("final_shape before transformer",final_nodes.shape)
    return final_nodes

def construct_brain_map(C, F_roi):
    """
    C: (B, 45, 54, 45) long
    F_roi: (B, 400, D)
    Returns:
        Q: (B, 45, 54, 45, D)
    """
    B, _, _, _ = C.shape
    D = F_roi.shape[-1]
    device = F_roi.device
    zero = torch.zeros(B, 1, D, device=device)
    F_pad = torch.cat([zero, F_roi], dim=1)  # (B, 401, D)
    batch_idx=torch.arange(B,device=device).view(B,1,1,1)
    Q=F_pad[batch_idx, C]

    return Q

class BrainTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, depth=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=0.4,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

    def forward(self, x):
        return self.encoder(x)


# --------------------------------------------------
# 5. Full Model
# --------------------------------------------------

class AtlasFreeBrainTransformer(nn.Module):
    def __init__(
        self,
        roi_feat_dim=512,
        embed_dim=360,
        num_heads=4,
        depth=2,
        num_classes=2
    ):
        super().__init__()
        self.ffn = FFN(
            in_dim=roi_feat_dim,
            out_dim=embed_dim
        )
        self.pool=nn.AvgPool3d(kernel_size=3, stride=2)

        self.transformer = BrainTransformer(
            dim=embed_dim,
            num_heads=num_heads,
            depth=depth
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(255, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )

    def forward(self, F_roi, C):
        """
        F_roi: (B, 400, 1632)
        C: (B, 45, 54, 45)
        """
        # print("Shape After applying PCA:",F_roi.shape)

        # 1. ROI embedding
        F_emb = self.ffn(F_roi)  # (B, 400, D)
        # print("Shape After passing through the FNN",F_emb.shape)

        # 2. Construct voxel brain map
        Q = construct_brain_map(C, F_emb)  # (B, 45, 54, 45, D)
        # print("shape After Constructing the brain map:",Q.shape)

        # 3. Block pooling
        Q=Q.permute(0,4,1,2,3)
        # print("Shape After the permute function:",Q.shape)
        # tokens=self.pool(Q)
        tokens=extract_nodes(Q, kernel_size=config['kernel_size'], stride=config['stride'])

        # 4. Transformer
        tokens = self.transformer(tokens)  # (B, N, D)

        # 5. Subject-level pooling
        h = tokens.mean(dim=1)  # (B, D)

        # 6. Classification
        out = self.classifier(h)
        return out