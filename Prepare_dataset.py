import os
import shutil
import argparse

def main():
    parser=argparse.ArgumentParser(description='Organize .mat files into folders')
    parser.add_argument('--datadir',
                        type=str,
                        default='.',
                        help=help='Path to the directory containing the .mat files (default: current directory)')
    args=parser.parse_args()

    source=args.datadir
    if not os.path.isdir(source):
        print(f"Error: Directory '{source}' does not exist!")
    features_dir=os.path.join(source,'Features')
    cluster_dir=os.path.join(source,'Cluster_index_mat')

    os.makedirs(features_dir,exist_ok=True)
    os.makedirs(cluster_dir,exist_ok=True)
    print(f"Processing directory: {source}")
    moved_count = 0
    
    # Single loop through all files
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        
        # Only process files (skip directories) and .mat files
        if not os.path.isfile(file_path) or not filename.endswith('.mat'):
            continue
            
        # Move feature files
        if filename.startswith('s_') and '_feature.mat' in filename:
            dest_path = os.path.join(features_dir, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved → Features: {filename}")
            moved_count += 1
            
        # Move cluster index files
        elif filename.startswith('s_') and '_cluster_index.mat' in filename:
            dest_path = os.path.join(cluster_dir, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved → Cluster_index_mat: {filename}")
            moved_count += 1
    
    print(f"\nDone! {moved_count} files were organized.")
    print(f"Features folder: {features_dir}")
    print(f"Cluster_index_mat folder: {cluster_dir}")
if __name__=="__main__":
    main()

