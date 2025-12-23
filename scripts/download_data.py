"""
Download IMDB dataset
"""
import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = Path("data/raw")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_imdb():
    """Download and extract IMDB dataset"""
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "aclImdb_v1.tar.gz"
    
    # Download
    if not tar_path.exists():
        print(f"📥 Downloading IMDB dataset...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(DATA_URL, tar_path, reporthook=t.update_to)
        print("✅ Download complete!")
    else:
        print("📦 Dataset already downloaded")
    
    # Extract
    extract_dir = DATA_DIR / "aclImdb"
    if not extract_dir.exists():
        print("📂 Extracting archive...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)
        print("✅ Extraction complete!")
    else:
        print("📂 Dataset already extracted")
    
    # Print stats
    train_pos = len(list((extract_dir / "train" / "pos").glob("*.txt")))
    train_neg = len(list((extract_dir / "train" / "neg").glob("*.txt")))
    test_pos = len(list((extract_dir / "test" / "pos").glob("*.txt")))
    test_neg = len(list((extract_dir / "test" / "neg").glob("*.txt")))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train: {train_pos + train_neg:,} reviews")
    print(f"   Test:  {test_pos + test_neg:,} reviews")
    print(f"   Total: {train_pos + train_neg + test_pos + test_neg:,} reviews")
    print(f"\n✅ Dataset ready at: {extract_dir}")

if __name__ == "__main__":
    download_imdb()