# Training Config file
##### DO NOT EDIT THESE LINES #####
config = {}
###################################
config['label_data_dir']="/content/Dataset_shuai_zip/label.mat"
config['feature_data_dir']="/content/Dataset_shuai_zip/Features"
config['cluster_data_dir']="/content/Dataset_shuai_zip/Cluster_index_mat/"
config['output_dir']="/content/output"
config['batch_size']=16
config['Epochs']=50
config['n_components']=512
config['use_pca']=False
config['learning_rate']=2e-4
config['weight_decay']=0.05
config['kernel_size']=5
config['stride']=5
config['n_split']=10