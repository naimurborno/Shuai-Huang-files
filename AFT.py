import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_pca import PCA
from ptdec.dec import DEC 
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
        return self.net(F_roi)  # (B, 400, out_dim)
    

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
    return final_nodes


def construct_brain_map(C, F_roi):
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



class AtlasFreeBrainTransformer(nn.Module):
    def __init__(
        self,
        roi_feat_dim=512,
        embed_dim=360,
        num_heads=4,
        depth=2,
        num_classes=2,
        encoder_hidden_size = 32,
        cluster_num = 50
    ):
        super().__init__()
        self.hidden_dim=360
        self.num_grid=5096
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
        self.pos_embed=nn.Parameter(torch.zeros(1, 1000, embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_grid, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, self.hidden_dim * self.num_grid),
        )
        self.dec = DEC(cluster_number=cluster_num, hidden_dimension=self.hidden_dim, encoder=self.encoder, orthogonal=True, freeze_center=True, project_assignment=True)

    def forward(self, F_roi, C):
        F_emb = self.ffn(F_roi)  # (B, 400, 512)-> (B, 400, 360) Passing Connectivity feature vector

        Q = construct_brain_map(C, F_emb)  # (B, 45, 54, 45, 360) Constructing Multi Channel Brain Map

        Q=Q.permute(0,4,1,2,3)

        tokens=extract_nodes(Q, kernel_size=config['kernel_size'], stride=config['stride']) # Extracting nodes using sum pooling to create Node features

        tokens = self.transformer(tokens)  # (B, N, D) #Multi Head Self Attention Transformer

        h ,_ = self.dec(tokens)  # (B, D) Subject Level Feature Vector
        h=h.mean(dim=1)
        out = self.classifier(h) # Classifier head
        
        return out