import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_pca import PCA
import train_config
config = train_config.config
class ROIEmbedding(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
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

    # pad background vector
    zero = torch.zeros(B, 1, D, device=device)
    F_pad = torch.cat([zero, F_roi], dim=1)  # (B, 401, D)
    # print(C.shape)
    batch_idx=torch.arange(B,device=device).view(B,1,1,1)
    Q=F_pad[batch_idx, C]
    # print(Q.shape)
    # print(D.shape)
    # indexing
    # Q = F_pad.gather(
    #     1,
    #     C.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    # )

    return Q


# --------------------------------------------------
# 3. 3D Block Pooling
# --------------------------------------------------

class BlockPooling(nn.Module):
    def __init__(self, block_size=3, stride=3):
        super().__init__()
        self.block_size = block_size
        self.stride = stride

    def forward(self, Q):
        # Q: (B, X, Y, Z, D)
        B, X, Y, Z, D = Q.shape
        # print()

        Q = Q.permute(0, 4, 1, 2, 3)  # (B, D, X, Y, Z)
        # print("the shape of Q:", Q.shape)

        Q = F.avg_pool3d(
            Q,
            kernel_size=self.block_size,
            stride=self.stride
        )
        # print("The shape of Q after pooling:",Q.shape)
        Q=Q*self.block_size*self.block_size*self.block_size

        Q = Q.flatten(2).transpose(1, 2)  # (B, num_blocks, D)
        return Q
# --------------------------------------------------
# 4. Transformer Encoder
# --------------------------------------------------

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
        embed_dim=128,
        num_heads=4,
        depth=2,
        num_classes=2
    ):
        super().__init__()
        self.roi_embed = ROIEmbedding(
            in_dim=roi_feat_dim,
            out_dim=embed_dim
        )
        self.pool=nn.AvgPool3d(kernel_size=3, stride=2)
        self.block_pool = BlockPooling()
        self.transformer = BrainTransformer(
            dim=embed_dim,
            num_heads=num_heads,
            depth=depth
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, F_roi, C):
        """
        F_roi: (B, 400, 1632)
        C: (B, 45, 54, 45)
        """
        # print("Shape After applying PCA:",F_roi.shape)

        # 1. ROI embedding
        F_emb = self.roi_embed(F_roi)  # (B, 400, D)
        # print("Shape After passing through the FNN",F_emb.shape)

        # 2. Construct voxel brain map
        Q = construct_brain_map(C, F_emb)  # (B, 45, 54, 45, D)
        # print("shape After Constructing the brain map:",Q.shape)

        # 3. Block pooling
        Q=Q.permute(0,4,1,2,3)
        # print("Shape After the permute function:",Q.shape)
        # tokens=self.pool(Q)
        tokens=extract_nodes(Q, kernel_size=config['kernel_size'], stride=config['stride'])
        # tokens = self.block_pool(Q)  # (B, N, D)
        # print("Shape after pooling function:",tokens.shape)
        # tokens=tokens.flatten(2)
        # print("shape after the flatten function:",tokens.shape)
        # tokens=tokens.transpose(1,2)
        # print("Shape After the transpose function and ready to be fet in transformer:",tokens.shape)

        # 4. Transformer
        tokens = self.transformer(tokens)  # (B, N, D)
        # print("Shape after transpose and after transformer section:",tokens.shape)

        # 5. Subject-level pooling
        h = tokens.mean(dim=1)  # (B, D)
        # print("shape after subject level pooling:", h.shape)

        # 6. Classification
        out = self.classifier(h)
        return out