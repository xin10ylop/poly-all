"""Build a 10-minute-grid test set (Aug 1+) and score it with the model trained through Jul 31 (walk-forward)."""
import os, sys, pickle
os.environ.setdefault('GRID_STEPS', '1'); os.environ.setdefault('DATE_MIN', '2026-08-01'); os.environ.setdefault('DS_OUT', 'data/stack_ds_10min.parquet')
sys.path.insert(0, os.path.dirname(__file__))
from stack import build, SF
df = build()
df = df.assign(kindH=(df.kind == 'high').astype(int))
df = df[df.ref.notna()]
clf = pickle.load(open('data/stack_model_2026-08-01.pkl', 'rb'))
df['p'] = clf.predict_proba(df[SF])[:, 1]
df.to_parquet('data/stack_test_10min.parquet')
print('rows', len(df))
