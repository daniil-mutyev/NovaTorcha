# src/train.py
import torch, sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from src.model import TinyGPT

SP_PATH = "tokenizer/sp.model"
MAX_LEN = 512
LR = 2e-4
EPOCHS = 1
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_TOP4 = "<TOP4>"
END_TOP4 = "</TOP4>"

class PairDataset(Dataset):
    def __init__(self, pairs, sp):
        self.pairs = pairs
        self.sp = sp
        self.start_id = sp.piece_to_id(START_TOP4)
        self.end_id   = sp.piece_to_id(END_TOP4)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        prompt, target = self.pairs[idx]
        text = (prompt + target).strip()
        ids = self.sp.encode(text)
        ids = ids[:MAX_LEN]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)

        # build loss mask: 1 only for tokens inside <TOP4> ... </TOP4>
        mask = torch.zeros_like(y, dtype=torch.bool)
        inside = False
        for i, tid in enumerate(y.tolist()):
            if tid == self.start_id:
                inside = True
                mask[i] = 0  # don’t learn to predict the start token itself
            elif tid == self.end_id:
                inside = False
                mask[i] = 1  # predict end token
            else:
                mask[i] = 1 if inside else 0

        return x, y, mask

def collate(batch):
    xs, ys, ms = zip(*batch)
    maxT = max(x.size(0) for x in xs)
    def pad(seq):
        out = torch.full((len(seq), maxT), 0)
        for i, t in enumerate(seq):
            out[i, :t.size(0)] = t
        return out
    return pad(xs), pad(ys), pad(ms)

def train_loop(pairs):
    sp = spm.SentencePieceProcessor(model_file=SP_PATH)
    ds = PairDataset(pairs, sp)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

    model = TinyGPT(vocab_size=sp.vocab_size()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        for x, y, m in pbar:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)

            logits = model(x)                   # (B,T,V)
            B, T, V = logits.shape
            loss_mask = m.view(B*T)
            loss = torch.nn.functional.cross_entropy(
                logits.view(B*T, V)[loss_mask],
                y.view(B*T)[loss_mask]
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/tinygpt.pt")
    print("Saved checkpoints/tinygpt.pt")

if __name__ == "__main__":
    # For now, expect you built pairs file earlier
    # Quick dummy example to verify pipeline:
    demo_pairs = [
        ("<CTX> age=30; city=A; </CTX>\n<PRODUCTS> ... </PRODUCTS>\nPlease output Top4 as: <TOP4> P1 | P2 | P3 | P4 </TOP4>\n",
         "<TOP4> TravelCard | PremiumCard | Brokerage | SavingsDeposit </TOP4>")
    ] * 64
    train_loop(demo_pairs)
