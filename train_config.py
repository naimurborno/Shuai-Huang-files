# Training Config file
# 2D Cycle GAN configuration file 

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######
config['label_data_dir']="/content/Dataset_shuai_zip/label.mat"
config['feature_data_dir']="/content/Dataset_shuai_zip/Features"
config['cluster_data_dir']="/content/Dataset_shuai_zip/Cluster_index_mat/"
config['batch_size']=16
config['Epochs']=50
config['n_components']=512
config['use_pca']=False