import os, json, time, pickle, hashlib, shutil
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pyseltongue import PlaintextToHexSecretSharer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('IP-Based Flows Pre-Processed Train.csv')
test_df  = pd.read_csv('IP-Based Flows Pre-Processed Test.csv')

print("✅ Train and test datasets loaded")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
#configs
LABEL_COL = "is_attack"             # name of label column in your train/test
CAT_COLS = ["traffic", "service"]   # categorical columns to encode (adjust if needed)
SHARD_SIZE = 3000                   # rows per shard; if None, NUM_SHARDS used
NUM_SHARDS = 5                      # used when SHARD_SIZE is None
SHAMIR_N = 5
SHAMIR_T = 3
NODE_IDS = [f"node{i}" for i in range(1, SHAMIR_N+1)]
STORAGE_ROOT = "3_p2p_storage"
LEDGER_PATH = "3_ledger.json"
TIMINGS_CSV = "3_timings.csv"
SUMMARY_JSON = "3_pipeline_output_summary.json"
REPORT_MD = "3_results_report.md"
MIN_SAMPLES_FOR_LOCAL_TRAIN = 5
SCALE_FEATURES = True
RANDOM_STATE = 42
#utility
def now(): return time.time()
def log_timing(stage, shard_id, start_ts, duration_s, notes=""):
    header = ["stage","shard_id","start_ts","duration_s","notes"]
    exists = os.path.exists(TIMINGS_CSV)
    with open(TIMINGS_CSV, "a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join([str(stage), str(shard_id), str(start_ts), f"{duration_s:.6f}", str(notes).replace(",",";")]) + "\n")
#preprocs
def preprocess(train_df, test_df, cat_cols, label_col, scale=True):
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    # validate
    if label_col not in train.columns or label_col not in test.columns:
        raise ValueError(f"Label column '{label_col}' missing in train/test.")
    for c in cat_cols:
        if c not in train.columns or c not in test.columns:
            raise ValueError(f"Categorical column '{c}' missing in train/test.")
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        train[c] = le.fit_transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
        encoders[c] = le
    X_train_df = train.drop(columns=[label_col])
    y_train = train[label_col].astype(int).reset_index(drop=True)
    X_test_df = test.drop(columns=[label_col])
    y_test = test[label_col].astype(int).reset_index(drop=True)
    feature_names = X_train_df.columns.tolist()
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df.values)
        X_test = scaler.transform(X_test_df.values)
    else:
        X_train = X_train_df.values
        X_test = X_test_df.values
    return X_train, y_train, X_test, y_test, feature_names, scaler, encoders
#shard const
def build_shards(X, y, shard_size=None, n_shards=5):
    n = len(y)
    if shard_size and shard_size > 0:
        shards = []
        for i in range(0, n, shard_size):
            Xs = X[i:i+shard_size]
            ys = y.iloc[i:i+shard_size].reset_index(drop=True)
            shards.append((Xs, ys))
        return shards
    else:
        n_shards = min(n_shards, n)
        skf = StratifiedKFold(n_splits=n_shards, shuffle=False)
        shards = []
        for _, test_idx in skf.split(X, y):
            Xs = X[test_idx]
            ys = y.iloc[test_idx].reset_index(drop=True)
            shards.append((Xs, ys))
        return shards
#enc -dec uisng aes-gcm
def encrypt_shard_obj(obj):
    key = AESGCM.generate_key(bit_length=256)
    aes = AESGCM(key)
    pt = pickle.dumps(obj)
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, pt, associated_data=None)
    sha = hashlib.sha256(ciphertext).hexdigest()
    # return key so caller can split it; do NOT write key to disk
    return {"key": key, "nonce": nonce, "ciphertext": ciphertext, "sha256": sha}

def decrypt_shard_obj(enc_blob, key):
    aes = AESGCM(key)
    pt = aes.decrypt(enc_blob["nonce"], enc_blob["ciphertext"], associated_data=None)
    return pickle.loads(pt)

# shamir secrt sharing
def split_key(key_bytes, n, t):
    hex_key = key_bytes.hex()
    shares = PlaintextToHexSecretSharer.split_secret(hex_key, t, n)
    return shares

def recover_key(shares):
    hex_key = PlaintextToHexSecretSharer.recover_secret(shares)
    return bytes.fromhex(hex_key)

#storage and merkle root
def ensure_storage(node_ids, storage_root):
    os.makedirs(storage_root, exist_ok=True)
    os.makedirs(os.path.join(storage_root,"shards"), exist_ok=True)
    for n in node_ids:
        os.makedirs(os.path.join(storage_root, n, "shards"), exist_ok=True)

def save_encrypted(enc_blob, shard_id, storage_root):
    path = os.path.join(storage_root, "shards", f"{shard_id}.enc")
    with open(path, "wb") as f:
        pickle.dump({"nonce":enc_blob["nonce"], "ciphertext":enc_blob["ciphertext"], "sha256":enc_blob["sha256"]}, f)
    return path

def distribute_shares_to_nodes(shares, shard_id, node_ids, storage_root):
    mapping = {}
    for i, s in enumerate(shares):
        node = node_ids[i % len(node_ids)]
        path = os.path.join(storage_root, node, f"{shard_id}.share")
        with open(path, "w") as f:
            f.write(s)
        mapping[node] = path
    return mapping

def replicate_enc(enc_path, shard_id, node_ids, storage_root):
    for n in node_ids:
        dst = os.path.join(storage_root, n, "shards", f"{shard_id}.enc")
        shutil.copy2(enc_path, dst)

def merkle_root_from_hex(hex_list):
    nodes = [bytes.fromhex(h) for h in hex_list]
    if not nodes:
        return ""
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        new = []
        for i in range(0, len(nodes), 2):
            new.append(hashlib.sha256(nodes[i] + nodes[i+1]).digest())
        nodes = new
    return nodes[0].hex()

def append_to_ledger(manifest, ledger_path):
    entry = {"ts": time.time(), "utc": datetime.utcnow().isoformat(), "manifest": manifest}
    ledger = []
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, "r") as f:
                ledger = json.load(f)
        except Exception:
            ledger = []
    ledger.append(entry)
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)
    return entry

#retreival and verify 
def collect_shares(shard_id, node_order, t, storage_root):
    shares = []
    for n in node_order:
        path = os.path.join(storage_root, n, f"{shard_id}.share")
        if os.path.exists(path):
            with open(path, "r") as f:
                shares.append(f.read().strip())
            if len(shares) >= t:
                break
    return shares

def find_encrypted_on_nodes(shard_id, node_order, storage_root):
    for n in node_order:
        p = os.path.join(storage_root, n, "shards", f"{shard_id}.enc")
        if os.path.exists(p):
            return p
    fallback = os.path.join(storage_root, "shards", f"{shard_id}.enc")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("Encrypted shard not found")

def retrieve_and_decrypt_shard(shard_id, node_order, t, storage_root):
    shares = collect_shares(shard_id, node_order, t, storage_root)
    if len(shares) < t:
        raise RuntimeError(f"Not enough shares ({len(shares)}) to recover key (need {t})")
    key = recover_key(shares)
    enc_path = find_encrypted_on_nodes(shard_id, node_order, storage_root)
    with open(enc_path, "rb") as f:
        enc_blob = pickle.load(f)
    if hashlib.sha256(enc_blob["ciphertext"]).hexdigest() != enc_blob["sha256"]:
        raise RuntimeError("SHA-256 mismatch: content tampered")
    return decrypt_shard_obj(enc_blob, key)

#local train and fed avg
def local_train(X, y):
    if len(y) < MIN_SAMPLES_FOR_LOCAL_TRAIN:
        return None
    m = LogisticRegression(max_iter=200)
    m.fit(X, y)
    return {"coef": m.coef_.astype(float), "intercept": m.intercept_.astype(float), "n": len(y)}

def fedavg(updates):
    updates = [u for u in updates if u is not None]
    if not updates:
        return None
    total = sum(u["n"] for u in updates)
    coef_sum = sum(u["coef"] * u["n"] for u in updates)
    intercept_sum = sum(u["intercept"] * u["n"] for u in updates)
    return {"coef": coef_sum / total, "intercept": intercept_sum / total}

#orchested 
def run_full_pipeline(train_df, test_df):
    # clean prior timing file
    if os.path.exists(TIMINGS_CSV):
        os.remove(TIMINGS_CSV)

    # 1. Preprocess
    t0 = now()
    X_train, y_train, X_test, y_test, features, scaler, encoders = preprocess(train_df, test_df, CAT_COLS, LABEL_COL, SCALE_FEATURES)
    log_timing("preprocess", "-", t0, now() - t0, f"features={len(features)},train={len(y_train)},test={len(y_test)}")
    print(f"[1] Preprocess done: features={len(features)}, train={len(y_train)}, test={len(y_test)}")

    # 2. Shard construction
    t0 = now()
    shards = build_shards(X_train, y_train, SHARD_SIZE, NUM_SHARDS)
    log_timing("shard_construct", "-", t0, now() - t0, f"num_shards={len(shards)},sizes={[len(s[1]) for s in shards]}")
    print(f"[2] Shards built: count={len(shards)}, sizes={[len(s[1]) for s in shards]}")

    ensure_storage(NODE_IDS, STORAGE_ROOT)

    manifest = {"created": datetime.utcnow().isoformat(), "shards": []}
    per_shard_summary = {}

    # 3. For each shard: encrypt, hash, split key, distribute shares, replicate
    for i, (Xs, ys) in enumerate(shards):
        shard_id = f"shard_{i}"
        # encrypt
        t0 = now()
        enc_blob = encrypt_shard_obj((Xs, ys))
        log_timing("encrypt", shard_id, t0, now() - t0, f"rows={len(ys)}")
        # save enc
        t0 = now()
        enc_path = save_encrypted(enc_blob, shard_id, STORAGE_ROOT)
        log_timing("save_enc", shard_id, t0, now() - t0, enc_path)
        # split key
        t0 = now()
        shares = split_key(enc_blob["key"], SHAMIR_N, SHAMIR_T)
        log_timing("shamir_split", shard_id, t0, now() - t0, f"n={SHAMIR_N},t={SHAMIR_T}")
        # distribute shares
        t0 = now()
        mapping = distribute_shares_to_nodes(shares, shard_id, NODE_IDS, STORAGE_ROOT)
        log_timing("distribute_shares", shard_id, t0, now() - t0, f"nodes={list(mapping.keys())}")
        # replicate
        t0 = now()
        replicate_enc(enc_path, shard_id, NODE_IDS, STORAGE_ROOT)
        log_timing("replicate", shard_id, t0, now() - t0, f"replicated_to={len(NODE_IDS)}")
        # manifest entry
        manifest["shards"].append({"shard_id": shard_id, "sha256": enc_blob["sha256"], "rows": len(ys), "enc_path": enc_path})
        per_shard_summary[shard_id] = {"rows": len(ys), "sha256": enc_blob["sha256"], "enc_path": enc_path, "nodes": list(mapping.keys())}
        print(f"    - shard {shard_id}: rows={len(ys)} sha256={enc_blob['sha256'][:16]}...")

    # 4. Merkle root + ledger append
    t0 = now()
    hex_list = [s["sha256"] for s in manifest["shards"]]
    mroot = merkle_root_from_hex(hex_list)
    manifest["merkle_root"] = mroot
    ledger_entry = append_to_ledger(manifest, LEDGER_PATH)
    log_timing("merkle_ledger", "-", t0, now() - t0, f"merkle_root={mroot}")
    print(f"[3] Merkle root computed and appended to ledger: {mroot}")

    # 5. Retrieval simulation, decrypt, verify, local train
    local_updates = []
    for s in manifest["shards"]:
        sid = s["shard_id"]
        t0 = now()
        try:
            Xs, ys = retrieve_and_decrypt_shard(sid, NODE_IDS, SHAMIR_T, STORAGE_ROOT)
            log_timing("retrieve_decrypt", sid, t0, now() - t0, f"rows={len(ys)}")
            print(f"    - retrieved & decrypted {sid}, rows={len(ys)}")
            t1 = now()
            up = local_train(Xs, ys)
            log_timing("local_train", sid, t1, now() - t1, f"trained={'yes' if up else 'no'},n={len(ys)}")
            if up:
                print(f"      local training done n={up['n']}")
            else:
                print(f"      local training skipped (not enough samples) n={len(ys)}")
            local_updates.append(up)
        except Exception as e:
            log_timing("retrieve_decrypt_fail", sid, t0, now() - t0, str(e))
            print(f"    - retrieval failed for {sid}: {e}")
            local_updates.append(None)

    # 6. FedAvg
    t0 = now()
    global_update = fedavg(local_updates)
    log_timing("fedavg", "-", t0, now() - t0, f"has_update={global_update is not None}")
    if global_update:
        print("[4] FedAvg produced global update (coef shape):", global_update["coef"].shape)
    else:
        print("[4] FedAvg produced no update (no local models)")

    # 7. Save summary JSON & results_report.md
    summary = {
        "features": len(features),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "num_shards": len(manifest["shards"]),
        "shards": per_shard_summary,
        "merkle_root": mroot,
        "ledger_entry": ledger_entry,
        "global_update_present": global_update is not None,
        "global_update": global_update,
        "global_update_coef": global_update["coef"].tolist() if global_update else None,
        "global_update_intercept": global_update["intercept"].tolist() if global_update else None
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # produce markdown report skeleton
    try:
        timings_df = pd.read_csv(TIMINGS_CSV)
    except Exception:
        timings_df = None

    md = []
    md.append("# Secure Shard Pipeline — Results Report\n")
    md.append(f"**Run (UTC):** {datetime.utcnow().isoformat()}\n")
    md.append("## Dataset\n")
    md.append(f"- Features: {len(features)}\n- Train samples: {len(y_train)}\n- Test samples: {len(y_test)}\n")
    md.append("## Shards\n")
    md.append("|shard_id|rows|sha256_prefix|enc_path|\n|---|---:|---|---|\n")
    for sid,info in per_shard_summary.items():
        md.append(f"|{sid}|{info['rows']}|{info['sha256'][:16]}...|{info['enc_path']}|\n")
    md.append("\n## Merkle & Ledger\n")
    md.append(f"- merkle_root: `{mroot}`\n- ledger entry ts: `{ledger_entry['ts']}`\n")
    md.append("\n## Timings (aggregate)\n")
    if timings_df is not None:
        agg = timings_df.groupby("stage")["duration_s"].agg(["count","sum","mean"]).reset_index()
        md.append("|stage|count|total_s|mean_s|\n|---|---:|---:|---:|\n")
        for _,r in agg.iterrows():
            md.append(f"|{r['stage']}|{int(r['count'])}|{r['sum']:.6f}|{r['mean']:.6f}|\n")
    else:
        md.append("- timings.csv missing or unreadable.\n")
    md.append("\n## FedAvg\n")
    if global_update:
        md.append(f"- global_update coef shape: {global_update['coef'].shape}\n")
    else:
        md.append("- no global update (no local training results)\n")
    md.append("\n## Artifacts\n")
    md.append(f"- Encrypted shards: `{STORAGE_ROOT}/shards/*.enc`\n- Node shares: `{STORAGE_ROOT}/node*/<shard>.share`\n- Ledger: `{LEDGER_PATH}`\n- Timings: `{TIMINGS_CSV}`\n- Summary JSON: `{SUMMARY_JSON}`\n")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md))

    print("[DONE] Pipeline finished.")
    print("Artifacts written:", TIMINGS_CSV, SUMMARY_JSON, REPORT_MD, LEDGER_PATH)
    return summary

if __name__ == "__main__":
    out = run_full_pipeline(train_df, test_df)