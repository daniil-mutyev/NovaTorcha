# src/infer.py
import torch, sentencepiece as spm
from src.model import TinyGPT

PRODUCTS = [
    "TravelCard","PremiumCard","CreditCard","FX","CashLoan",
    "MultiFXDeposit","TimeDeposit","SavingsDeposit","Brokerage","GoldBars"
]

SP_PATH = "tokenizer/sp.model"
CKPT = "checkpoints/tinygpt.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_top4(prompt: str) -> list[str]:
    sp = spm.SentencePieceProcessor(model_file=SP_PATH)
    model = TinyGPT(vocab_size=sp.vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    ids = sp.encode(prompt)[:510]  # leave space
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # token ids
    prod_ids = {sp.piece_to_id(p) for p in PRODUCTS}
    bar_id   = sp.piece_to_id("|")
    end_id   = sp.piece_to_id("</TOP4>")

    used = set()
    allow = set(prod_ids) | {bar_id, end_id}

    with torch.no_grad():
        for _ in range(64):
            logits = model(x)[:, -1, :]  # (1,V)
            mask = torch.full_like(logits, float("-inf"))
            mask[:, list(allow)] = 0.0
            # no duplicates
            for pid in used:
                logits[0, pid] = float("-inf")
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
            nid = next_id.item()
            x = torch.cat([x, next_id], dim=1)

            if nid in prod_ids:
                used.add(nid)
                if len(used) == 4:
                    # force end next
                    allow = {end_id}
            if nid == end_id:
                break

    names = [sp.id_to_piece(pid) for pid in used]
    # Order approximation: we append greedily; x’s order after <TOP4> may be read back from sequence.
    # For now just return set order:
    return names[:4]
