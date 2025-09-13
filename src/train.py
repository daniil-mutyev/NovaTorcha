# src/train.py
import torch
from torch.utils.data import DataLoader, Dataset
from src.model import TinyGPT

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}

def main():
    vocab_size = 8000  # placeholder
    model = TinyGPT(vocab_size)
    print(model)

if __name__ == "__main__":
    main()
