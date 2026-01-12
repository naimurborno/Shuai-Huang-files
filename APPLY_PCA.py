import torch
def apply_pca(features,pca_model,train_data=False):
    if train_data==True:
        assert features.dim()==3
        B, T, F=features.shape
        features=features.view(B*T, F)
        F_roi_2d=pca_model.transform(features)
        F_roi_2d=F_roi_2d.to(torch.float32)
        F_roi=F_roi_2d.view(B, T, -1)
        return F_roi
    else:
        assert features.dim()==3
        B, T, F=features.shape
        features=features.view(B*T, F)
        F_roi_2d=pca_model.transform(features)
        F_roi_2d=F_roi_2d.to(torch.float32)
        F_roi=F_roi_2d.view(B,T,-1)
        return F_roi
