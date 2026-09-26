"""Train the production stacked model on ALL available data (rolling retrain; validated by rolling.sh)."""
import sys, os, pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from stack import SF
df = pd.read_parquet('data/stack_ds.parquet')
df = df.assign(kindH=(df.kind == 'high').astype(int))
tr = df[df.ref.notna()]
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                     l2_regularization=1.0, random_state=0).fit(tr[SF], tr.y)
pickle.dump(clf, open('data/stack_model_prod.pkl', 'wb'))
print('trained on', len(tr), 'rows through', tr.date.max())
