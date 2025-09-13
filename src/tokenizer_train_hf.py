# src/tokenizer_train.py
import os, sentencepiece as spm

SPECIAL_TOKENS = [
    "<CTX>", "</CTX>", "<PRODUCTS>", "</PRODUCTS>",
    "<TOP4>", "</TOP4>", "|"
]
# Make product names single tokens (easier generation)
PRODUCTS = [
    "TravelCard","PremiumCard","CreditCard","FX","CashLoan",
    "MultiFXDeposit","TimeDeposit","SavingsDeposit","Brokerage","GoldBars"
]

def train_tokenizer(corpus_path: str, out_dir: str = "tokenizer", vocab_size: int = 8000):
    os.makedirs(out_dir, exist_ok=True)
    user_tokens = SPECIAL_TOKENS + PRODUCTS
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=os.path.join(out_dir, "sp"),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=user_tokens,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    print(f"Tokenizer saved to {out_dir}/sp.model")

if __name__ == "__main__":
    # Provide a simple seed corpus (you can point to a larger text file later)
    # A tiny corpus still works because we add user-defined tokens.
    with open("tokenizer_seed.txt","w",encoding="utf-8") as f:
        f.write(" ".join(SPECIAL_TOKENS + PRODUCTS))
    train_tokenizer("tokenizer_seed.txt")
