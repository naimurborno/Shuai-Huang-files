from torch_pca import PCA
import torch
def apply_pca(features,n_components=512):
    assert features.dim()==3
    B, T, F=features.shape
    features=features.view(B*T, F)
    pca_model=PCA(n_components=n_components)
    F_roi_2d=pca_model.fit_transform(features)
    F_roi=F_roi_2d.view(B, T, -1)
    return F_roi