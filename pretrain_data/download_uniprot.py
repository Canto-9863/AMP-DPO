import os
import requests
import gzip
import shutil
from tqdm import tqdm
import argparse

def download_file(url, local_path, chunk_size=8192):
    """下载文件，显示进度条"""
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_path)) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return local_path

def extract_gz(gz_path, extract_path):
    """解压缩.gz文件"""
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return extract_path

def main():
    parser = argparse.ArgumentParser(description='下载UniProt数据')
    parser.add_argument('--dataset', type=str, default='swiss', choices=['swiss', 'trembl', 'uniref50', 'uniref90'],
                       help='要下载的数据集类型')
    parser.add_argument('--output_dir', type=str, default='data/uniprot',
                       help='输出目录')
    args = parser.parse_args()
    
    # 数据来源URLs
    urls = {
        'swiss': 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',
        'trembl': 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',
        'uniref50': 'https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz',
        'uniref90': 'https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz'
    }
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载数据
    url = urls[args.dataset]
    file_name = os.path.basename(url)
    local_gz_path = os.path.join(args.output_dir, file_name)
    
    print(f"下载 {args.dataset} 数据...")
    download_file(url, local_gz_path)
    
    # 解压数据
    extract_path = local_gz_path.replace('.gz', '')
    print(f"解压 {file_name}...")
    extract_gz(local_gz_path, extract_path)
    
    print(f"下载和解压完成: {extract_path}")

if __name__ == '__main__':
    main()