# src/data_prep.py
import pandas as pd
from pathlib import Path
from typing import List, Tuple

PRODUCTS = [
    "TravelCard","PremiumCard","CreditCard","FX","CashLoan",
    "MultiFXDeposit","TimeDeposit","SavingsDeposit","Brokerage","GoldBars"
]

def _fmt_float(x):
    try:
        return f"{float(x):.4g}"
    except Exception:
        return "0"

def render_prompt(row: pd.Series) -> str:
    return (
        "<CTX>\n"
        f"age={row.get('age','')}; city={row.get('city','')}; status={row.get('status','')}; "
        f"avg_balance={_fmt_float(row.get('avg_monthly_balance_KZT',0))}; "
        f"total_spend_3m={_fmt_float(row.get('total_spend',0))}; monthly_spend={_fmt_float(row.get('monthly_spend',0))};\n"
        f"travel_ratio={_fmt_float(row.get('travel_ratio',0))}; online_ratio={_fmt_float(row.get('online_ratio',0))}; "
        f"restaurant_ratio={_fmt_float(row.get('restaurant_ratio',0))}; fx_ratio={_fmt_float(row.get('fx_ratio',0))}; "
        f"jewelry_ratio={_fmt_float(row.get('jewelry_ratio',0))};\n"
        f"atm_freq={_fmt_float(row.get('atm_freq',0))}; loan_freq={_fmt_float(row.get('loan_freq',0))}; "
        f"cc_freq={_fmt_float(row.get('cc_freq',0))}; fx_activity={int(row.get('fx_activity',0))};\n"
        f"inflow={_fmt_float(row.get('inflow',0))}; outflow={_fmt_float(row.get('outflow',0))}; "
        f"net_cash_flow={_fmt_float(row.get('net_cash_flow',0))}; free_balance={_fmt_float(row.get('free_balance',0))}; "
        f"spend_volatility={_fmt_float(row.get('spend_volatility',0))}\n"
        "</CTX>\n"
        "<PRODUCTS> " + " | ".join(PRODUCTS) + " </PRODUCTS>\n"
        "Please output Top4 as: <TOP4> P1 | P2 | P3 | P4 </TOP4>\n"
    )

def render_target(top4: List[str]) -> str:
    # top4 must contain 4 strings chosen from PRODUCTS
    return "<TOP4> " + " | ".join(top4) + " </TOP4>"

def build_pairs_from_df(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> List[Tuple[str,str]]:
    """
    features_df: one row per client with engineered numeric fields already.
    labels_df: columns: client_code, p1, p2, p3, p4 (product names matching PRODUCTS)
    returns list of (prompt_text, target_text)
    """
    pairs = []
    merged = features_df.merge(labels_df, on="client_code", how="inner")
    for _, row in merged.iterrows():
        prompt = render_prompt(row)
        target = render_target([row["p1"], row["p2"], row["p3"], row["p4"]])
        pairs.append((prompt, target))
    return pairs

def dump_corpus(pairs, out_path="data/corpus.txt"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt, target in pairs:
            f.write(prompt.strip()+"\n")
            f.write(target.strip()+"\n\n")
    print(f"Saved {len(pairs)} examples to {out_path}")
