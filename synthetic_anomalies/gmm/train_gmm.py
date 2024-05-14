import sys
sys.path.append("../..")

from data.data_loader import normalised_subspace

from sklearn.mixture import GaussianMixture
from joblib import dump

''' add entire fitting procedure here in one pipeline'''

X = normalised_subspace()

n_components = 71

gmm = GaussianMixture(n_components, covariance_type='full', max_iter=1000)

gmm.fit(X)

dump(gmm, './gmm.joblib')
