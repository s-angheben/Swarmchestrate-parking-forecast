"""Train one model on data/processed/<city>/, evaluate it on test, save it in checkpoints/<city>_<model>[_type]/.

    uv run ai_component/train.py --city chania --model wide [--with_type] [--k 10] [--batch_size 32] [--epochs 10]

Models (24 h in, 2 h out, a call at every 15-min bin, MSE on all data: failures as the API sees them):
- wide: one iTransformer over all sensors (events and occupancy of every sensor in and out)
- sensor_cluster: sensors grouped once (k-means on each sensor's average day over train); the model forecasts the k cluster means
- behavior_cluster: k-means centroids from the train sensor-days; at every call each sensor joins the nearest centroid from
  its last 24 h; the model forecasts the k cluster means (empty clusters: 0, left out of the loss)
--with_type: also use the sensor type (wide: in the column embedding; cluster models: type one-hot added to the k-means
features, weighing as much as the events); saved as <city>_<model>_type.
Cluster models: each sensor gets its cluster's forecast. Test metrics per sensor and city, on all data and on working
sensors only (mask = 1).
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import iTransformerWide
from utils.timefeatures import time_features                          # Time-Series-Library (path set by model.py)
import clusters

SEQ_LEN, PRED_LEN = 96, 8                                              # 24 h in, 2 h out

p = argparse.ArgumentParser()
p.add_argument("--city", default="chania")
p.add_argument("--model", default="wide", choices=["wide", "sensor_cluster", "behavior_cluster"])
p.add_argument("--k", type=int, default=10, help="clusters (cluster models)")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--epochs", type=int, default=10, help="max epochs (early stopping after 3 without improvement)")
p.add_argument("--with_type", action="store_true", help="also use the sensor type")
args = p.parse_args()
DATA = Path("data/processed") / args.city
OUT = Path("checkpoints") / f"{args.city}_{args.model}{'_type' if args.with_type else ''}"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
wide = args.model == "wide"

# data: bins x sensors (sensors sorted by id), the three splits together so val/test inputs can use the previous 24 h
d = pd.concat([pd.read_parquet(DATA / f"{s}.parquet", columns=["time", "id", "events", "occupancy", "mask"]).assign(split=s)
               for s in ["train", "val", "test"]])
events, occupancy, mask = (d.pivot(index="time", columns="id", values=n) for n in ["events", "occupancy", "mask"])
split = d.drop_duplicates("time").set_index("time").split.reindex(events.index).values
sensors = events.columns
S = len(sensors)
E, O, Mk = (x.values.astype(np.float32) for x in (events, occupancy, mask))
hours = events.index.hour.values
tr = split == "train"

# static information of each sensor: type index (and one-hot) and standardised location (+ missing flag)
meta = pd.read_parquet(DATA / "test.parquet", columns=["id", "type", "latitude", "longitude"]).drop_duplicates("id").set_index("id").loc[sensors]
type_name = meta.type.fillna("unknown")                                 # sensors without metadata: own type
types = sorted(type_name.unique())
sensor_type = type_name.map({t: i for i, t in enumerate(types)}).values
onehot = np.eye(len(types))[sensor_type]                               # (S, types)
add_type = (lambda F: np.hstack([F, onehot])) if args.with_type else (lambda F: F)   # cluster features (+ type)
if not args.with_type:
    sensor_type = np.zeros(S, int)                                      # wide: one type for all = type off
lat, lon = meta.latitude.values, meta.longitude.values
missing = np.isnan(lat)
location = np.stack([np.where(missing, 0.0, (lat - np.nanmean(lat)) / np.nanstd(lat)),
                     np.where(missing, 0.0, (lon - np.nanmean(lon)) / np.nanstd(lon)), missing], axis=1).astype(np.float32)

# clusters: label_at(c) = cluster of each sensor for a call at bin c (None: wide)
K = S if wide else args.k
if args.model == "sensor_cluster":
    F = add_type(clusters.day_features(E[tr], O[tr], hours[tr]))        # each sensor's average day over train
    centroids, scale = clusters.fit(F, K)
    fixed = clusters.assign(F, centroids, scale)
    label_at = lambda c: fixed
    print("sensors per cluster:", np.bincount(fixed, minlength=K).tolist())
elif args.model == "behavior_cluster":
    t0, t1 = np.where(tr)[0][[0, -1]]
    F = np.concatenate([add_type(clusters.day_features(E[b:b + 96], O[b:b + 96], hours[b:b + 96])) for b in range(t0, t1 - 94, 96)])
    centroids, scale = clusters.fit(F, K)                               # on the train sensor-days
    L = np.zeros((len(E), S), np.int64)
    for c in range(SEQ_LEN, len(E)):
        L[c] = clusters.assign(add_type(clusters.day_features(E[c - 96:c], O[c - 96:c], hours[c - 96:c])), centroids, scale)
    label_at = lambda c: L[c]
    n = np.stack([np.bincount(L[c], minlength=K) for c in range(SEQ_LEN, len(E))])
    print("mean sensors per cluster:", n.mean(0).round(1).tolist(), f"| empty: {(n == 0).mean():.1%} of cluster-calls")
else:
    label_at = lambda c: None

# scaling: one mean and std per target, over all sensors and train bins
mu, sd = np.array([E[tr].mean(), O[tr].mean()], np.float32), np.array([E[tr].std(), O[tr].std()], np.float32)
mu2, sd2 = np.repeat(mu, K), np.repeat(sd, K)
TF = time_features(events.index, freq="15min").T.astype(np.float32)   # (bins, 5) calendar features, local time


def series(a, b, lab):
    """Scaled events and occupancy of bins [a, b): per sensor (lab None) or per cluster; and 1 for the columns with data."""
    e, o, n = E[a:b], O[a:b], np.ones(S)
    if lab is not None:
        (e, n), (o, _) = clusters.pool(e, lab, K), clusters.pool(o, lab, K)
    return ((np.concatenate([e, o], 1) - mu2) / sd2).astype(np.float32), np.tile(n > 0, 2).astype(np.float32)


class Windows(Dataset):
    """One sample per call inside a split: input [c-96, c), calendar, output [c, c+8) and its loss weight."""
    def __init__(self, part):
        idx = np.where(split == part)[0]
        self.calls = np.arange(max(idx[0], SEQ_LEN), idx[-1] - PRED_LEN + 2)
    def __len__(self):
        return len(self.calls)
    def __getitem__(self, i):
        c = self.calls[i]
        lab = label_at(c)
        x, _ = series(c - SEQ_LEN, c, lab)
        y, w = series(c, c + PRED_LEN, lab)
        return x, TF[c - SEQ_LEN:c], y, np.broadcast_to(w, y.shape).copy()


data = {s: Windows(s) for s in ["train", "val", "test"]}
cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=SEQ_LEN, pred_len=PRED_LEN,
                      e_layers=4, d_model=512, d_ff=512, n_heads=8, dropout=0.1,
                      factor=1, activation="gelu", embed="timeF", freq="t", output_attention=False,
                      enc_in=2 * K, num_class=1, n_types=len(types) if wide and args.with_type else 1,
                      with_type=args.with_type, types=types if args.with_type else None,
                      model=args.model, k=None if wide else K, tz=str(events.index.tz),   # tz: clock of the bins
                      batch_size=args.batch_size, learning_rate=1e-4, lradj="type1", train_epochs=args.epochs, patience=3,
                      loss="MSE (empty clusters left out)", seed=0)
json.dump(vars(cfg), open(OUT / "config.json", "w"), indent=2)
json.dump({"mean": mu.tolist(), "std": sd.tolist(), "targets": ["events", "occupancy"]}, open(OUT / "scaler.json", "w"), indent=2)
meta.to_csv(OUT / "sensors.csv")                                       # the sensors of the model (order) + metadata
if args.model == "sensor_cluster":
    pd.DataFrame({"id": sensors, "cluster": fixed}).to_csv(OUT / "clusters.csv", index=False)
if args.model == "behavior_cluster":
    json.dump({"centroids": centroids.tolist(), "scale": scale.tolist()}, open(OUT / "centroids.json", "w"))
print(f"{args.model}: {S} sensors, {K} columns, windows { {s: len(w) for s, w in data.items()} }, device {device}")

# training: Adam, learning rate halved every epoch, best model on validation kept, early stopping
torch.manual_seed(cfg.seed)
model = iTransformerWide(cfg, sensor_type, location) if wide else iTransformerWide(cfg, np.zeros(K, int), np.zeros((K, 3)))
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data["val"], batch_size=args.batch_size)


def run_epoch(loader, train):
    model.train(train)
    total, n = 0.0, 0.0
    with torch.set_grad_enabled(train):
        for x, xm, y, w in loader:
            x, xm, y, w = x.to(device), xm.to(device), y.to(device), w.to(device)
            loss = (((model(x, xm) - y) ** 2) * w).sum() / w.sum()           # MSE, empty clusters left out
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * w.sum().item(); n += w.sum().item()
    return total / n


log, best, wait = [], np.inf, 0
for epoch in range(1, cfg.train_epochs + 1):
    lr = cfg.learning_rate * 0.5 ** (epoch - 1)
    for g in opt.param_groups:
        g["lr"] = lr
    t0 = time.time()
    train_loss, val_loss = run_epoch(train_loader, True), run_epoch(val_loader, False)
    log.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, "seconds": round(time.time() - t0, 1)})
    pd.DataFrame(log).to_csv(OUT / "log.csv", index=False)
    print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.1e} | {log[-1]['seconds']} s", flush=True)
    if val_loss < best:
        best, wait = val_loss, 0
        torch.save(model.state_dict(), OUT / "model.pt")
    else:
        wait += 1
        if wait >= cfg.patience:
            print("early stopping"); break

# test: best model; each sensor gets its column (wide) or its cluster's forecast; scaled back and clipped
model.load_state_dict(torch.load(OUT / "model.pt", map_location=device))
model.eval()
with torch.no_grad():
    P = np.concatenate([model(x.to(device), xm.to(device)).cpu().numpy()
                        for x, xm, _, _ in DataLoader(data["test"], batch_size=args.batch_size)])
P = P * sd2 + mu2                                                      # (calls, 8, 2K), real units
calls = data["test"].calls
lab = np.stack([np.arange(S) if wide else label_at(c) for c in calls])[:, None, :]   # (calls, 1, S)
pred = {"events": np.take_along_axis(P[..., :K], lab, 2).clip(0), "occupancy": np.take_along_axis(P[..., K:], lab, 2).clip(0, 1)}
true = {"events": np.stack([E[c:c + PRED_LEN] for c in calls]), "occupancy": np.stack([O[c:c + PRED_LEN] for c in calls])}
masks = {"all": np.ones_like(true["events"]), "working": np.stack([Mk[c:c + PRED_LEN] for c in calls])}


def masked_metrics(p, y, m):
    """p, y, m: (calls, steps, sensors). MAE and RMSE overall and per step, only where m == 1."""
    err, n = (p - y) * m, m.sum(axis=(0, 2))
    cols = ["overall"] + [f"+{15 * (h + 1)} min" for h in range(p.shape[1])]
    return pd.DataFrame([[np.abs(err).sum() / m.sum()] + list(np.abs(err).sum(axis=(0, 2)) / n),
                         [np.sqrt((err ** 2).sum() / m.sum())] + list(np.sqrt((err ** 2).sum(axis=(0, 2)) / n))],
                        index=["MAE", "RMSE"], columns=cols)


def city_avg(a, m):                                                    # average over the sensors in m: (calls, 8, 1)
    return ((a * m).sum(axis=2) / m.sum(axis=2))[..., None]


one = np.ones((len(calls), PRED_LEN, 1))
metrics = pd.concat({**{(k, "per sensor", t): masked_metrics(pred[t], true[t], m) for k, m in masks.items() for t in pred},
                     **{(k, "city", t): masked_metrics(city_avg(pred[t], m), city_avg(true[t], m), one) for k, m in masks.items() for t in pred}},
                    names=["data", "level", "target", "metric"])
metrics.to_csv(OUT / "metrics.csv")
print(f"test: {len(calls)} calls")
print(metrics["overall"].unstack(["target", "metric"]).round(4).to_string())
print("saved in", OUT)
