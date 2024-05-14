import sys
sys.path.append("..")

import copy
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
#import pickle
import joblib
from data.data_loader import scaler, subspace, loader
import os

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

path, _ = os.path.split(os.path.realpath(__file__))

filename = f"{path}/gmm/gmm.joblib"

class AnomalyGenerator(): # could inherit from GaussianMixture
    def __init__(self, X = None):
        self.X = loader()[subspace].to_numpy()
        self.n_components = 71 # Found via. BIC fitting

        try:
            self.model = joblib.load(open(filename, 'rb'))
        except:
            print("Read model file failed.")
            self.model = GaussianMixture(self.n_components, covariance_type='full', max_iter=1000)
            self.model.fit(X)

        self.local_gmm = None
        self.cluster_gmm = None
        self.noise_gmm = None


    def alpha_vector(self, lower, upper):
        a = np.ones(5)
        a[np.random.randint(4)] += np.random.uniform(lower, upper)
        return a
    
    def alpha_matrix(self, lower, upper, leading = True):
        '''
        Used for generating scaling factors for anomaly production
        '''
        
        diagonal_matrices = np.zeros((self.n_components, len(self.X[0]), len(self.X[0])))
        diagonal_matrices[:, np.arange(len(self.X[0])), np.arange(len(self.X[0]))] = np.random.uniform(lower, upper, size=(self.n_components, len(self.X[0])))
        
        if leading:
            return diagonal_matrices
        else:
            return np.flip(diagonal_matrices, axis=1)
        
    def cluster(self, n = 200, n_clusters = 10, alpha = 1.2):
        clusters = []
        params = self.model._get_parameters()

        for i in range(n_clusters):    
            idx = np.random.randint(len(params[0])) # select random cluster

            while (np.sqrt(np.diag(params[2][idx])) > .05).sum() >= 1: # Select from normal clusters
                idx = np.random.randint(len(params[0]))

            clusters.append(np.random.multivariate_normal(
                mean = alpha*params[1][idx], # add half a std to each cluster mean - mean shift
                cov = .1*params[2][idx], # I want to have tighter clusters - this can be changed.
                check_valid="ignore",
                size = int(n / n_clusters)
            ))
        return np.concatenate(clusters)
    
    def local(self, n, alpha = 16, lower = None, upper = None, n_clusters = 10): # Draw from 5 sigma distributions - check this
        if not self.local_gmm:
            self.local_gmm = copy.deepcopy(self.model)
        
        params = self.model._get_parameters()

        #alpha = np.diag([alpha for alpha in range(5)])
   
        self.local_gmm._set_parameters(
            ( # clean up
                params[0], # weights
                params[1], # means
                alpha*params[2], # covariances - 3 sigma
                params[3], # precisions_cholesky
            )
        )

        '''locals_ = []

        for i in range(n_clusters):    
            idx = np.random.randint(len(params[0])) # select random cluster

            while (np.sqrt(np.diag(params[2][idx])) > .2).sum() >= 1: # Select from most normal clusters - relax
                idx = np.random.randint(len(params[0]))

            locals_.append(np.hstack((np.random.multivariate_normal(
                mean = params[1][idx], # add half a std to each cluster mean - mean shift
                cov = alpha*params[2][idx], # I want to have tighter clusters - this can be changed.
                check_valid="ignore",
                size = int(n / n_clusters)
            ), idx*np.ones(int(n / n_clusters)).reshape(-1, 1))))'''

        
        samples = self.local_gmm.sample(n)
        return np.hstack((samples[0], samples[1].reshape(-1,1))) #np.concatenate(locals_) 
    
    def _global(self, n, normal, alpha = 3):
        '''
            Draw from a random uniform distribution with limits of 4 sigma of the means of each feature.
            Could be some changes to how this is done but the scores are on the level of subtly that I want.
        '''
        # should I filter the scores of these values so that only values below threshold are included?
        # ie below min of generated samples.
        return np.random.uniform(
            low = normal.mean(axis=0) - alpha*normal.std(axis=0), #[i/alpha if i > 0 else alpha*i for i in self.X.min(axis=0)],
            high = normal.mean(axis=0) + alpha*normal.std(axis=0),
            size=(n, normal.shape[1])
        )
    
    def gaussian_noise(self, n = 200, noise_level = .25): # How much do I degrade performance by.
        params = self.model._get_parameters()
        if not self.noise_gmm:
            self.noise_gmm = copy.deepcopy(self.model)

        self.noise_gmm._set_parameters(
            (
                params[0], # weights
                0*params[1], # means 
                noise_level*params[2], # covariances
                params[3], # precisions_cholesky
            )
        )
        return self.noise_gmm.sample(n)

    def generate_anomaly_dataset(self, anom_size = 3000):
        '''
        '''
        normal = np.hstack((self.X, np.zeros((len(self.X),1 ))))

        locals_ = np.hstack((self.local(
            anom_size
        ), np.ones((anom_size, 1 ))))

        globals_ = np.hstack((self._global(
            anom_size
        ), 2*np.ones((anom_size, 1))))
    
        clusters_ = np.hstack((self.cluster(
            n = anom_size
        ), 3*np.ones((anom_size, 1))))
        '''print(locals_.shape, clusters_.shape)'''
        anoms = np.concatenate([
            normal, 
            locals_, 
            globals_, 
            clusters_
        ])

        return anoms
    
def physical_filter(data):

    X = scaler.inverse_transform(data[:, 0:5])

    a_condition1 = X[:, 2] >= 0.3
    a_condition2 = X[:, 2] < 6

    e_condition1 = X[:, 4] >= 0
    e_condition2 = X[:, 4] <= 1.5

    sini_condition1 = X[:, 3] >= -1.0
    sini_condition2 = X[:, 3] <= 1.0

    chained_conditions = (a_condition1) & (a_condition2) & (e_condition1) & (e_condition2) & (sini_condition1) & (sini_condition2)

    return np.hstack((scaler.transform(X[chained_conditions]), data[chained_conditions, 5].reshape(-1, 1)))

def normal_filter(df):
    '''
    Sigma clipping of anomalies.
    Filter any mutations that fall within in normal range (i.e. 3sigma of normal scores.)
    '''

    #grouping = df.groupby(by="label")
    norm_mean = df[df["label"] == 0]["score"].mean()
    norm_std = df[df["label"] == 0]["score"].std()

    return df[~((df["score"] >= (norm_mean - 3*norm_std[0])) & (df["label"] != 0))]

def create_data_segment(samples, anom_gen, label):
    samples = physical_filter(samples)
    #print("samples before hstack:")
    #print(samples[0])
    samples = np.hstack((samples[:, 0:5], label*np.ones((len(samples), 1)), anom_gen.model.score_samples(samples[:, 0:5]).reshape(-1, 1), samples[:, 5].reshape(-1, 1)))
    #print("samples after hstack:")
    #print(samples[0])
    return samples

def anomaly_dataset(anom_size = 200):
    anom_gen = AnomalyGenerator()

    X_norm = scaler.transform(anom_gen.X)

    scores = anom_gen.model.score_samples(X_norm)

    normal_samples = X_norm[~((scores) < (np.mean(scores) - 3 * np.std(scores))), :]

    normal = create_data_segment(
        np.hstack((normal_samples[:50000, :], -1*np.ones(50000).reshape(-1, 1))), 
        anom_gen, 0
    )

    globals_ = create_data_segment(
        np.hstack((anom_gen._global(n = anom_size, normal = normal), -1*np.ones(anom_size).reshape(anom_size, 1))), 
        anom_gen, 1) # Globals have no cluster membership so return -1
    local_anoms = anom_gen.local(n = anom_size)
    
    locals_ = create_data_segment(local_anoms, anom_gen, 2) # local_anoms[1] IS CLUSTER membership label.
    clusters = create_data_segment(np.hstack((anom_gen.cluster(n = anom_size), -1*np.ones(anom_size).reshape(anom_size, 1))), anom_gen, 3) # Cluster anoms should no belong to any clusters in GMM so return -1

    anoms = np.concatenate([
        normal, 
        locals_, 
        globals_, 
        clusters
    ])

    anom_df = pd.DataFrame(
        columns=[*subspace, "label", "score", "cluster_label"],
        data = np.hstack((scaler.inverse_transform(anoms[:, 0:5]), anoms[:, [5,6,7]])) # 5=label, 6=score ,7=cluster label
    )

    regl = abs(anom_df["score"] - anom_df["score"].max()) # Regularize

    regl /= regl.max()

    anom_df["rank"] = regl

    gaussian_noise = anom_gen.gaussian_noise(
            n = len(anom_df), noise_level=1
    )[0]

    for i, feature in enumerate(subspace):
        anom_df[f"{feature}_noise"] = gaussian_noise[:, i]

    output = anom_df #.sort_values(by="rank", ascending=False)
    #output = output[(output["label"] == 0) & (output["score"] < (output["score"].mean() - 4*output["score"].std()))] # Sigma clip any extremes from the normal sample of data 

    return output


'''
scores = anom_gen.model.score_samples(anom_gen.X)
    normal_samples = anom_gen.X[~((scores) < (np.mean(scores) - 3 * np.std(scores))), :][:50000, :]#anom_gen.model.sample(50000)
    normal = create_data_segment(
        np.hstack((normal_samples[0], normal_samples[1].reshape(len(normal_samples[1]), 1))), 
        anom_gen, 0
    )

anom_gen = AnomalyGenerator()
X = anom_gen.X

scores = anom_gen.model.score_samples(anom_gen.X)
normal_samples = anom_gen.X[~((scores) < (np.mean(scores) - 3 * np.std(scores))), :][:50000, :]#anom_gen.model.sample(50000)
print(normal_samples.shape)

normal = create_data_segment(
    np.hstack((normal_samples, -1*np.ones(50000).reshape(-1, 1))), 
    anom_gen, 0
)
'''

''''''
'''

print(anomaly_dataset(anom_size = 200))
anom_gen = AnomalyGenerator()
x = anom_gen.local(n=300)
print(x.shape)

print(anom_gen.model.score_samples(x[:, 0:5]))'''

'''
anom_gen = AnomalyGenerator()
print(anom_gen)

anom_gen.local(n=300)
'''
## Fix sini problem - multiple values can't determine high or low inclination - redo GMM fit to fix this.

'''
print(f"Normal: {norm_scores.min(), norm_scores.max(), norm_scores.mean(), norm_scores.std(), len(norm_scores)}")
print(glob_scores.min(), glob_scores.max(), glob_scores.mean(), glob_scores.std(), len(glob_scores))
print(local_scores.min(), local_scores.max(), local_scores.mean(), local_scores.std(), len(local_scores))
print(clusters_scores.min(), clusters_scores.max(), clusters_scores.mean(), clusters_scores.std(), len(clusters_scores))'''
'''
anom_gen = AnomalyGenerator()

normal = physical_filter(anom_gen.model.gmm.sample(50000)[0])
normal = np.hstack((normal, np.zeros((len(normal), 1)), anom_gen.model.gmm.score_samples(normal[:, 0:5]).reshape(len(normal), 1)))
#normal = np.hstack((normal, anom_gen.model.gmm.score_samples(normal[:, 0:5]).reshape(len(normal), 1)))

#normal = np.hstack((normal))
#print(anom_gen.model.gmm.score_samples(normal[:, 0:5]))
print(normal)'''
