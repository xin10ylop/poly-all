"""Nowcast model for Elon Musk post-count bracket markets (resolution: xtracker.polymarket.com).

At time t inside a window [s, e): C = posts with createdAt >= s and importedAt <= t (what we could know).
Remaining count N_rem ~ NegBin(mean mu, size r), mu = sum over remaining hours of rate * diurnal[hour_ET].
rate = blend of trailing 24h / 72h / 7d posting rates (posts per 'diurnal-weighted hour').
"""
import json
import numpy as np, pandas as pd
from scipy.stats import nbinom

ET = 'America/New_York'


def load_posts(path='data/elon_posts.json'):
    P = pd.DataFrame(json.load(open(path))['data'])
    P['ct'] = (pd.to_datetime(P.createdAt, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    P['it'] = (pd.to_datetime(P.importedAt, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    P['it'] = np.maximum(P.it, P.ct)
    P.loc[P.it - P.ct > 3600, 'it'] = P.ct + 300     # backfilled imports: assume normal 5-min lag
    return P.sort_values('ct').reset_index(drop=True)


class Nowcast:
    def __init__(self, P, w24=0.3, w72=0.3, w7d=0.4, r=12.0, prof_days=28):
        self.ct = P.ct.values; self.it = P.it.values
        self.w = (w24, w72, w7d); self.r = r; self.prof_days = prof_days
        self.hour_et = pd.to_datetime(self.ct, unit='s', utc=True).tz_convert(ET).hour.values

    def diurnal(self, t):
        """Share of posts per ET hour over trailing prof_days (only posts known by t)."""
        m = (self.it <= t) & (self.ct >= t - self.prof_days * 86400)
        h = np.bincount(self.hour_et[m], minlength=24).astype(float) + 1.0
        return h / h.sum() * 24          # mean 1 per hour

    def count(self, s, t):
        return int(((self.ct >= s) & (self.it <= t)).sum())

    def rate(self, t, prof):
        """Posts per profile-weighted hour, trailing windows (known by t)."""
        out = []
        hrs = pd.date_range(pd.Timestamp(t - 7 * 86400, unit='s', tz='UTC'), periods=7 * 24, freq='h').tz_convert(ET).hour.values
        for days in (1, 3, 7):
            n = ((self.ct >= t - days * 86400) & (self.it <= t)).sum()
            wh = prof[hrs[-days * 24:]].sum()
            out.append(n / max(wh, 1e-6))
        return float(np.dot(self.w, out))

    def mu_remaining(self, t, e, prof, rate):
        hrs = pd.date_range(pd.Timestamp(t, unit='s', tz='UTC'), pd.Timestamp(e, unit='s', tz='UTC'), freq='h',
                            inclusive='left').tz_convert(ET).hour.values
        if len(hrs) == 0:
            return 0.0
        frac_first = 1.0
        return float(rate * prof[hrs].sum() * frac_first)

    def dist(self, s, t, e):
        """Return (C, mu, pmf over N_rem as array)."""
        prof = self.diurnal(t); rate = self.rate(t, prof)
        C = self.count(s, t)
        # posts created in [s, t) but not yet imported still count -> expected small: add rate * lag
        mu = self.mu_remaining(t, e, prof, rate) + rate * 150 / 3600
        return C, mu

    def bucket_probs(self, s, t, e, buckets):
        C, mu = self.dist(s, t, e)
        r = self.r
        p = r / (r + mu) if mu > 0 else 1.0
        out = []
        for lo, hi in buckets:
            a = max(lo - C, 0); b = hi - C
            if b < 0:
                out.append(0.0); continue
            if mu <= 0:
                out.append(1.0 if a == 0 else 0.0); continue
            out.append(float(nbinom.cdf(b, r, p) - (nbinom.cdf(a - 1, r, p) if a > 0 else 0.0)))
        return C, mu, np.array(out)
