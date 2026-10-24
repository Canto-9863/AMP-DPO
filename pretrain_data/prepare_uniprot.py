import os
import json
import argparse
from download_uniprot import download_file, extract_gz
from process_uniprot import process_fasta, merge_chunks
from preprocess_for_training import preprocess_for_training

def main():
    parser = argparse.ArgumentParser(description='完整UniProt数据准备流程')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--step', type=int, default=0,
                       help='起始步骤 (1: 下载, 2: 处理, 3: 准备训练)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    start_step = args.step
    
    if start_step <= 1:
        print("=== 步骤 1: 下载数据 ===")
        urls = {
            'swiss': 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',
            'trembl': 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',
            'uniref50': 'https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz',
            'uniref90': 'https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz'
        }
        
        dataset = config['dataset']
        download_dir = config['download_dir']
        
        os.makedirs(download_dir, exist_ok=True)
        
        url = urls[dataset]
        file_name = os.path.basename(url)
        local_gz_path = os.path.join(download_dir, file_name)
        
        print(f"下载 {dataset} 数据...")
        download_file(url, local_gz_path)
        
        extract_path = local_gz_path.replace('.gz', '')
        print(f"解压 {file_name}...")
        extract_gz(local_gz_path, extract_path)
        
        print(f"下载和解压完成: {extract_path}")
        fasta_path = extract_path
    else:
        dataset = config['dataset']
        download_dir = config['download_dir']
        fasta_path = os.path.join(download_dir, f"uniprot_{dataset}.fasta")
        if not os.path.exists(fasta_path):
            fasta_path = os.path.join(download_dir, f"{dataset}.fasta")
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"找不到FASTA文件，请检查路径或完成下载步骤: {fasta_path}")
        print(f"使用已有FASTA文件: {fasta_path}")
    
    if start_step <= 2:
        print("\n=== 步骤 2: 处理FASTA数据 ===")
        processed_dir = config['processed_dir']
        chunks_dir = os.path.join(processed_dir, 'chunks')
        
        process_fasta(
            fasta_path, 
            chunks_dir, 
            max_length=config['max_length'], 
            min_length=config['min_length'],
            chunk_size=config['chunk_size']
        )
        
        final_output = os.path.join(processed_dir, 'protein_sequences.csv')
        merge_chunks(
            chunks_dir, 
            final_output,
            sample_ratio=config['sample_ratio'],
            max_samples=config['max_samples'],
            seed=config['seed']
        )
        
        processed_csv = final_output
    else:
        processed_dir = config['processed_dir']
        processed_csv = os.path.join(processed_dir, 'protein_sequences.csv')
        if not os.path.exists(processed_csv):
            raise FileNotFoundError(f"找不到处理后的CSV文件，请检查路径或完成处理步骤: {processed_csv}")
        print(f"使用已有处理后数据: {processed_csv}")
    
    if start_step <= 3:
        print("\n=== 步骤 3: 准备训练数据 ===")
        training_dir = config['training_dir']
        
        preprocess_for_training(
            processed_csv,
            training_dir,
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            seed=config['seed']
        )
    
    print("\n全部准备工作完成！")
    print(f"训练数据路径: {config['training_dir']}/train")
    print(f"验证数据路径: {config['training_dir']}/val")
    print(f"测试数据路径: {config['training_dir']}/test")

if __name__ == '__main__':
    main()