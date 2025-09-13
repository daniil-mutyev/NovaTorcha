# src/make_pairs.py
import argparse, json
from pathlib import Path
import pandas as pd
from collections import defaultdict

PRODUCTS = [
    "TravelCard","PremiumCard","CreditCard","FX","CashLoan",
    "MultiFXDeposit","TimeDeposit","SavingsDeposit","Brokerage","GoldBars"
]

def _fmt(x):
    try: return f"{float(x):.4g}"
    except: return "0"

def build_features(clients, tx, tr):
    # ---- transactions ----
    tx["amount"] = pd.to_numeric(tx.get("amount", 0), errors="coerce").fillna(0.0)
    tx["category"] = tx.get("category", "").astype(str)
    tx["currency"] = tx.get("currency", "").astype(str)

    spend = tx.groupby("client_code")["amount"].sum().rename("total_spend")
    by_cat = tx.pivot_table(index="client_code", columns="category", values="amount", aggfunc="sum").fillna(0.0)
    eurusd = tx[tx["currency"].isin(["EUR","USD"])].groupby("client_code")["amount"].sum().rename("fx_amt")

    # ratios
    feat = pd.DataFrame(spend).join(eurusd, how="left").fillna(0.0)
    feat["monthly_spend"] = feat["total_spend"] / 3.0
    def ratio(cols):
        s = by_cat[cols].sum(axis=1) if set(cols).issubset(by_cat.columns) else 0.0
        return (s / feat["total_spend"]).fillna(0.0)

    feat["travel_ratio"]     = ratio(["Путешествия","Отели","Такси"])
    feat["online_ratio"]     = ratio(["Едим дома","Смотрим дома","Играем дома"])
    feat["restaurant_ratio"] = ratio(["Кафе и рестораны"])
    feat["jewelry_ratio"]    = ratio(["Ювелирное","Косметика"])
    feat["fx_ratio"]         = (feat["fx_amt"] / feat["total_spend"]).fillna(0.0)

    # ---- transfers ----
    tr["amount"] = pd.to_numeric(tr.get("amount", 0), errors="coerce").fillna(0.0)
    tr["type"] = tr.get("type", "").astype(str)
    tr["direction"] = tr.get("direction", "").astype(str)

    inflow  = tr[tr["direction"].str.lower().eq("in")].groupby("client_code")["amount"].sum().rename("inflow")
    outflow = tr[tr["direction"].str.lower().eq("out")].groupby("client_code")["amount"].sum().rename("outflow")
    cnt     = tr.groupby("client_code")["type"].count().rename("tr_cnt").replace(0, 1)

    def freq(t): 
        s = tr[tr["type"].eq(t)].groupby("client_code")["type"].count()
        return (s / cnt).fillna(0.0)

    f = pd.DataFrame(inflow).join(outflow, how="outer").fillna(0.0)
    f["atm_freq"]  = freq("atm_withdrawal")
    f["loan_freq"] = freq("loan_payment_out")
    f["cc_freq"]   = freq("cc_repayment_out")
    f["fx_activity"] = tr["type"].isin(["fx_buy","fx_sell"]).groupby(tr["client_code"]).any().astype(int)

    # ---- merge with clients ----
    clients["avg_monthly_balance_KZT"] = pd.to_numeric(
        clients.get("avg_monthly_balance_KZT", 0), errors="coerce"
    ).fillna(0.0)
    feat = clients.set_index("client_code").join(feat, how="left").join(f, how="left").fillna(0.0)

    # derived
    feat["net_cash_flow"] = feat["inflow"] - feat["outflow"]
    feat["free_balance"]  = feat["avg_monthly_balance_KZT"] - feat["monthly_spend"]

    # simple volatility proxy by month (optional quick version -> 0)
    feat["spend_volatility"] = 0.0
    return feat.reset_index()

def score_products(feat_row):
    # quick bootstrap rules (weights from your spec, simplified)
    s = defaultdict(float)
    s["TravelCard"]      = 0.7*feat_row.travel_ratio + 0.2*feat_row.fx_ratio + 0.1*0
    s["PremiumCard"]     = 0.5*_norm(feat_row.avg_monthly_balance_KZT) + 0.3*feat_row.restaurant_ratio + 0.2*feat_row.atm_freq
    s["CreditCard"]      = 0.4*0 + 0.4*feat_row.online_ratio + 0.2*feat_row.cc_freq
    s["FX"]              = 0.8*feat_row.fx_ratio + 0.2*feat_row.fx_activity
    deficit = max(0.0, feat_row.outflow - feat_row.inflow)
    s["CashLoan"]        = 0.5*_norm(deficit) + 0.3*feat_row.loan_freq + 0.2*(1 - _norm(feat_row.avg_monthly_balance_KZT))
    s["MultiFXDeposit"]  = 0.5*_norm(feat_row.avg_monthly_balance_KZT) + 0.3*feat_row.fx_ratio + 0.2*feat_row.travel_ratio
    s["TimeDeposit"]     = 0.7*_norm(feat_row.avg_monthly_balance_KZT) + 0.3*(1 - _norm(feat_row.spend_volatility))
    s["SavingsDeposit"]  = 0.6*_norm(feat_row.free_balance) + 0.4*0   # inflow_freq omitted
    s["Brokerage"]       = 0.6*_norm(feat_row.avg_monthly_balance_KZT) + 0.4*(1 if feat_row.net_cash_flow >= 0 else 0)
    s["GoldBars"]        = 0.5*feat_row.jewelry_ratio + 0.5*_norm(feat_row.avg_monthly_balance_KZT)
    return s

# very rough min-max to [0..1] without global stats; okay for bootstrap labels
def _norm(x): 
    x = float(x)
    return max(0.0, min(1.0, x / (abs(x) + 1e5)))  # squashing proxy

def render_prompt(r):
    return (
        "<CTX>\n"
        f"age={r.get('age','')}; city={r.get('city','')}; status={r.get('status','')}; "
        f"avg_balance={_fmt(r.get('avg_monthly_balance_KZT',0))}; total_spend_3m={_fmt(r.get('total_spend',0))}; "
        f"monthly_spend={_fmt(r.get('monthly_spend',0))};\n"
        f"travel_ratio={_fmt(r.get('travel_ratio',0))}; online_ratio={_fmt(r.get('online_ratio',0))}; "
        f"restaurant_ratio={_fmt(r.get('restaurant_ratio',0))}; fx_ratio={_fmt(r.get('fx_ratio',0))}; "
        f"jewelry_ratio={_fmt(r.get('jewelry_ratio',0))};\n"
        f"atm_freq={_fmt(r.get('atm_freq',0))}; loan_freq={_fmt(r.get('loan_freq',0))}; "
        f"cc_freq={_fmt(r.get('cc_freq',0))}; fx_activity={int(r.get('fx_activity',0))};\n"
        f"inflow={_fmt(r.get('inflow',0))}; outflow={_fmt(r.get('outflow',0))}; "
        f"net_cash_flow={_fmt(r.get('net_cash_flow',0))}; free_balance={_fmt(r.get('free_balance',0))}; "
        f"spend_volatility={_fmt(r.get('spend_volatility',0))}\n"
        "</CTX>\n"
        "<PRODUCTS> " + " | ".join(PRODUCTS) + " </PRODUCTS>\n"
        "Please output Top4 as: <TOP4> P1 | P2 | P3 | P4 </TOP4>\n"
    )

def render_target(p4): return "<TOP4> " + " | ".join(p4) + " </TOP4>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", required=True)
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--transfers", required=True)
    ap.add_argument("--out", default="data/pairs.jsonl")
    args = ap.parse_args()

    clients = pd.read_csv(args.clients)
    tx = pd.read_csv(args.transactions)
    tr = pd.read_csv(args.transfers)

    features = build_features(clients, tx, tr)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in features.iterrows():
            scores = score_products(r)
            top4 = sorted(scores, key=scores.get, reverse=True)[:4]
            prompt = render_prompt(r)
            target = render_target(top4)
            f.write(json.dumps({"prompt": prompt, "target": target}, ensure_ascii=False) + "\n")

    print(f"Wrote pairs → {args.out}, rows={len(features)}")

if __name__ == "__main__":
    main()
