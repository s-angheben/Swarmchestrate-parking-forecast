"""Raw parking messages (id, ts, oc) -> 15-min tables: events, occupancy, mask.

Rules:
- duplicates (same id and ts) are ACK retries: dropped
- silence: no message for more than 3 days = sensor failure; also before the first message
- events = state changes per bin (a message with the same state as the previous one is not counted)
- occupancy = fraction of the bin the spot is occupied (state carried forward from each message)
- mask = 0 for bins fully inside a silence (no data); there events and occupancy are 0
"""
import numpy as np
import pandas as pd

TZ = "Europe/Athens"
BIN = pd.Timedelta("15min")
# data provider: the devices communicate at least once a day,
# so no communication for more than 3 days = sensor failure
SILENCE = pd.Timedelta("3D")


def load_csv(path):
    """Raw export: id, ts (unix seconds), oc (true / false)."""
    m = pd.read_csv(path)
    m["ts"] = pd.to_datetime(m.ts, unit="s", utc=True)
    return m[["id", "ts", "oc"]]


def clean(messages, end):
    """Drop ACK retries (same id and ts), sort by sensor and time, keep messages before `end`."""
    m = messages.drop_duplicates(["id", "ts"]).sort_values(["id", "ts"], ignore_index=True)
    return m[m.ts < end].reset_index(drop=True)


def data_end(messages):
    """Start of the incomplete last bin = end of the last full bin."""
    return messages.ts.max().floor(BIN).tz_convert(TZ)        # floor in UTC: same bins as local time (offsets are whole hours)


def silences(m, start, end):
    """No-data periods (id, start, end): before the first message (from `start`), and between two messages
    or after the last message (until `end`) when longer than SILENCE. `m` = cleaned messages."""
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    g = m.groupby("id").ts
    before = g.min().rename("end").reset_index().assign(start=start)
    between = pd.DataFrame({"id": m.id, "start": g.shift(), "end": m.ts})
    after = g.max().rename("start").reset_index().assign(end=end)
    parts = [before[before.end > before.start], between[between.end - between.start > SILENCE], after[after.end - after.start > SILENCE]]
    return pd.concat(parts, ignore_index=True)[["id", "start", "end"]]


def drop_repeats(m):
    """Keep the first message of each sensor and the state changes (drop the 2nd of two equal states)."""
    return m[m.groupby("id").oc.shift() != m.oc].reset_index(drop=True)


def tables(m, start, end, sensors):
    """events, occupancy, mask: one row per bin in [start, end), one column per sensor in `sensors` (fixed order).
    `m` = cleaned messages (before `end`; messages before `start` are used for the state at `start`)."""
    bins = pd.date_range(start, end, freq=BIN, inclusive="left")
    b = bins.asi8                                                          # bin starts, ns since epoch (UTC)

    mask = pd.DataFrame(1, index=bins, columns=sensors, dtype="int8")
    col = {s: i for i, s in enumerate(sensors)}
    for sid, s0, s1 in silences(m, start, end).itertuples(index=False):   # bins fully inside a silence
        if sid in col:
            i0, i1 = np.searchsorted(b, s0.value), np.searchsorted(b, (s1 - BIN).value, side="right")
            mask.iloc[i0:i1, col[sid]] = 0
    mask.loc[:, ~mask.columns.isin(m.id)] = 0                            # sensor without any message: no data

    changes = drop_repeats(m)
    x = changes[changes.ts >= start]
    events = pd.crosstab(x.ts.dt.floor(BIN).dt.tz_convert(TZ), x.id) if len(x) else pd.DataFrame(dtype="int64")   # crosstab fails when empty
    events = events.reindex(index=bins, columns=sensors, fill_value=0)

    edges = bins.append(pd.DatetimeIndex([bins[-1] + BIN])).as_unit("s").asi8        # bin edges, seconds
    occupancy = pd.DataFrame(0.0, index=bins, columns=sensors)
    for sid, g in changes[changes.id.isin(sensors)].groupby("id"):
        t, s = g.ts.dt.as_unit("s").astype("int64").to_numpy(), g.oc.to_numpy(float)  # message times (s) and states
        cum = np.concatenate([[0.0], np.cumsum(s[:-1] * np.diff(t))])               # occupied seconds up to each message
        i = np.searchsorted(t, edges, side="right") - 1                             # last message at or before each edge
        j = i.clip(0)
        F = np.where(i >= 0, cum[j] + s[j] * (edges - t[j]), 0.0)                   # occupied seconds up to each edge
        occupancy[sid] = np.diff(F) / BIN.total_seconds()
    occupancy = occupancy.where(mask == 1, 0.0)
    return events, occupancy, mask


def window_tables(messages, state, t, sensors):
    """API call at time `t` (on a bin edge): tables of the 96 bins (24 h) before `t`.
    `state` (id, oc): state of every sensor at the start of the window (t - 24 h); `messages` (id, ts, oc): the messages
    of the window. No failed sensor, so the mask is always 1."""
    start = t - 96 * BIN                                                # absolute 24 h: always 96 bins, also on DST days
    m = messages[(messages.ts >= start) & (messages.ts < t)][["id", "ts", "oc"]]
    init = pd.DataFrame({"id": sensors, "ts": start.tz_convert("UTC") - pd.Timedelta("1s"),
                         "oc": state.set_index("id").oc.reindex(sensors).astype(bool).values})
    m = pd.concat([init, m]).sort_values(["id", "ts"], ignore_index=True)   # init: before the window, not an event
    return tables(m, start, t, sensors)
