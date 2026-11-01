"""nx idea G: event-mass correction. Global row AUC depends on WHICH events contain a click; V53's rank blend leaves its per-event
mass weak (event AUC ~0.563). Keep V53's within-event shares s = p / sum_event(p) and replace the event mass by a blend of
log(V53 mass) and a dedicated event-click model q(event has a click).

ONE code path, --day 5 / 6 / 7 (cutoff = START + day*DAY). Event model: LightGBM on label-free/causal user-behaviour features,
trained on training events with time < cutoff (labels < cutoff only), scored on the cohort (or evaluation) events.
Features of event e (user u, time t) use only: earlier events of u (time < t, all < cutoff for labels), the supplied history
(stamps < t) and the event's own 10 candidates (no labels).
Combination: log mass' = log m_V53 + w * z(logit q); w chosen on day 6, applied unchanged to day 5.
"""
import argparse, json, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R
from nx_seq_core import weighted_auc, cutoff_of, START, DAY

NAMES = ['n_prev', 'n_prev_click', 'rate_prev', 'ewm_prev', 'last_click', 'last3', 'gap_prev_ev', 'gap_prev_click_ev', 'n_today',
         'hist_len', 'gap_last_hist', 'hist_24h', 'hist_6h', 'hist_span', 'n_in_hist', 'max_rep', 'hour', 'warm_hist_only',
         'gap_prev_ev_log_ratio']


def user_event_features(heads, hc, hist, htimes, cands, cut):
    """heads: (E,7) event rows sorted by time; hc: event has click (only used for events < cut, enforced by caller via masking)."""
    u = heads[:, 0].astype(np.int64); t = heads[:, 2].astype(np.int64); E = len(u)
    lab = np.where(t < cut, hc.astype(np.float64), np.nan)  # labels at/after cutoff never visible
    o = np.lexsort((t, u)); us, ts, ls = u[o], t[o], lab[o]
    first = np.r_[True, us[1:] != us[:-1]]; grp = np.cumsum(first) - 1; gstart = np.flatnonzero(first)
    idx = np.arange(E) - gstart[grp]  # number of previous events of this user (all earlier in time)
    lk = np.nan_to_num(ls); known = ~np.isnan(ls)
    ck = np.cumsum(lk) - lk; cn = np.cumsum(known) - known; base = np.r_[0, np.cumsum(lk)][gstart][grp]; basen = np.r_[0, np.cumsum(known)][gstart][grp]
    n_prev_click = ck - base; n_known = cn - basen
    F = np.zeros((E, len(NAMES)))
    F[o, 0] = idx; F[o, 1] = n_prev_click; F[o, 2] = (n_prev_click + .63 * 3) / (n_known + 3)
    # ewm, last click, last-3 rate, gaps: sequential pass in numpy over lags
    ewm = np.full(E, .63); last = np.full(E, np.nan); l3s = np.zeros(E); l3n = np.zeros(E); gprev = np.full(E, 1e7); gclick = np.full(E, 1e7)
    for lag in range(1, 31):
        ok = idx >= lag; j = np.flatnonzero(ok); p = j - lag; lv = ls[p]; kn = ~np.isnan(lv)
        if lag == 1:
            gprev[j] = ts[j] - ts[p]; last[j] = lv
        if lag <= 3: l3s[j[kn]] += lv[kn]; l3n[j[kn]] += 1
        m = kn & (lv == 1) & (gclick[j] > 9e6); gclick[j[m]] = ts[j[m]] - ts[p[m]]
    # ewm via weights .7**lag over known labels
    num = np.zeros(E); den = np.zeros(E)
    for lag in range(1, 31):
        ok = idx >= lag; j = np.flatnonzero(ok); lv = ls[j - lag]; kn = ~np.isnan(lv); w = .7 ** lag
        num[j[kn]] += w * lv[kn]; den[j[kn]] += w
    ewm = (num + .63 * .5) / (den + .5)
    day0 = (ts - START) // DAY; ntoday = np.zeros(E)
    for lag in range(1, 31):
        ok = idx >= lag; j = np.flatnonzero(ok); m = day0[j - lag] == day0[j]; ntoday[j[m]] += 1
    F[o, 3] = ewm; F[o, 4] = np.nan_to_num(last, nan=-1); F[o, 5] = np.where(l3n > 0, l3s / np.maximum(l3n, 1), -1)
    F[o, 6] = np.log1p(gprev); F[o, 7] = np.log1p(gclick); F[o, 8] = ntoday
    v = hist >= 0; hl = v.sum(1); lastt = np.where(v, htimes, 0).max(1); firstt = np.where(v, htimes, np.iinfo(np.int64).max).min(1)
    F[:, 9] = hl; F[:, 10] = np.where(hl > 0, np.log1p(t - lastt), 20.); F[:, 11] = (v & (htimes >= t[:, None] - DAY)).sum(1)
    F[:, 12] = (v & (htimes >= t[:, None] - 6 * 3600)).sum(1); F[:, 13] = np.where(hl > 1, np.log1p(lastt - firstt), -1)
    inh = (cands[:, :, None] == np.where(v, hist, -2)[:, None, :]); F[:, 14] = inh.any(2).sum(1); F[:, 15] = inh.sum(2).max(1)
    F[:, 16] = (t % DAY) / 3600.; F[:, 17] = (hl > 0) & (F[:, 0] == 0)
    F[:, 18] = F[:, 6] - F[:, 10]
    return F


def run(day, seed=5501):
    tic = time.monotonic(); cut = cutoff_of(day)
    t = np.load(R / 'train.npy', mmap_mode='r'); heads = np.asarray(t[::10]); clk = np.asarray(t[:, 5]).reshape(-1, 10)
    hc = clk.sum(1) > 0; cand = np.asarray(t[:, 1]).reshape(-1, 10).astype(np.int64)
    hist = np.load(R / 'train_hist.npy', mmap_mode='r'); htim = np.load(R / 'train_times.npy', mmap_mode='r')
    ntr = int(np.searchsorted(heads[:, 2], cut))
    if day == 7:
        ev = np.load(R / 'evaluation.npy'); eh = np.asarray(np.load(R / 'evaluation_hist.npy')); et = np.asarray(np.load(R / 'evaluation_times.npy'))
        allh = np.concatenate([heads[:ntr], ev[::10]]); allc = np.concatenate([cand[:ntr], ev[:, 1].reshape(-1, 10).astype(np.int64)])
        allhist = np.concatenate([np.asarray(hist[:ntr]), eh]); allht = np.concatenate([np.asarray(htim[:ntr]), et]).astype(np.int64)
        allhc = np.concatenate([hc[:ntr], np.zeros(len(ev) // 10, bool)]); score_ix = np.arange(ntr, len(allh)); z = None
    else:
        z = np.load(R / f'v53_causal_propensity_day{day}.npz'); ei = z['events'][::10]
        assert heads[ei, 2].min() >= cut and heads[ei, 2].max() < cut + DAY
        keep = np.r_[np.arange(ntr), ei]  # pre-cutoff events + cohort events (other same-day events are not visible at evaluation)
        allh = heads[keep]; allc = cand[keep]; allhist = np.asarray(hist[keep]); allht = np.asarray(htim[keep]).astype(np.int64); allhc = hc[keep]
        score_ix = np.arange(ntr, len(keep))
    o = np.argsort(allh[:, 2], kind='stable'); assert np.all(o[:ntr] == np.arange(ntr))
    F = user_event_features(allh, allhc, allhist, allht, allc, cut)
    # train on pre-cutoff events in the last 4 days before cutoff (all labels < cutoff)
    tr = np.flatnonzero(allh[:ntr, 2] >= cut - 4 * DAY)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=63, min_child_samples=200, subsample=.8, subsample_freq=1,
                           colsample_bytree=.8, random_state=seed, verbose=-1)
    m.fit(F[tr], allhc[tr]); q = m.predict_proba(F[score_ix])[:, 1]
    info = dict(day=day, cutoff=int(cut), train_events=int(len(tr)), elapsed_s=round(time.monotonic() - tic, 1),
                importance=dict(zip(NAMES, m.booster_.feature_importance('gain').round(0).tolist())))
    return z, q, info


def combine(z, q, w):
    P = z['pred'].reshape(-1, 10); m = P.sum(1); s = P / m[:, None]; lq = np.log(q / (1 - q)); zq = (lq - lq.mean()) / lq.std()
    return (s * np.exp(np.log(m) + w * zq)[:, None]).ravel()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--day', type=int, nargs='+', default=[6, 5]); a = ap.parse_args()
    rep = {}
    for day in a.day:
        z, q, info = run(day)
        if z is None:
            np.save(R / 'nx_event_mass_eval_q.npy', q); print(json.dumps(info)); continue
        y, ev = z['y'], z['events']; hc = y.reshape(-1, 10).sum(1) > 0; base = weighted_auc(ev, y, z['pred'])
        m = z['pred'].reshape(-1, 10).sum(1)
        info.update(event_auc_q=float(roc_auc_score(hc, q)), event_auc_v53_mass=float(roc_auc_score(hc, m)), v53_adjusted=base, grid={})
        for w in (0, .1, .2, .3, .4, .5, .7, 1., 1.5):
            p = combine(z, q, w); info['grid'][w] = dict(adjusted=weighted_auc(ev, y, p), gain=weighted_auc(ev, y, p) - base, plain=float(roc_auc_score(y, p)))
        np.savez_compressed(R / f'nx_event_mass_day{day}.npz', q=q, events=ev[::10])
        rep[day] = info; print(json.dumps(info, indent=1), flush=True)
    (R / 'nx_event_mass_report.json').write_text(json.dumps(rep, indent=1, default=str))
