from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

def algo_results(data, models, label, iterations = 10, with_noise=True, noise_level = 1, silent = False):
    scores = {key : {
        "BAL_ACC": [],
        "F1": []
    } for key in models}

    for name, clf in zip(models.keys(), models.values()):
            
        for i in range(iterations):
            if with_noise:
                #duplicate_idx = np.random.choice(500, size=100, replace=False)

                noisy_features = data[:, :5] + data[:, -5:] # 3 sigma noise
                noisy_features =  np.concatenate([noisy_features]) #, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

                noisy_labels = np.concatenate([data[:, 5]]) #, data[duplicate_idx, 5]])
                
                # Should I make use of training score?
                
                X_train, X_test, y_train, y_test = train_test_split(
                    noisy_features, 
                    noisy_labels, 
                    test_size=.3,
                    #random_state=42,
                    stratify=noisy_labels,
                    shuffle=True
                )
                
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    data[:, 0:5], 
                    data[:, 5], 
                    test_size=.3,
                    #random_state=42,
                    stratify=data[:, 5],
                    shuffle=True
                )

            clf.fit(X_train)

            y_test_pred = clf.predict(X_test)  
            #y_test_scores = clf.decision_function(X_test) 
            
            f1 = f1_score(np.array(y_test)/label, y_test_pred)
            balanced_accuracy =  balanced_accuracy_score(y_test, label*y_test_pred)
            
            scores[name]["F1"].append(f1)
            scores[name]["BAL_ACC"].append(balanced_accuracy)
            
        if not silent:
            print(name)
            print(f"Accuracy metric: {scores[name]})")
            print(("------"))

    return scores


def algo_rank_results(data, models, label, iterations = 10, with_noise=True, noise_level = 0, silent = False):
    rank_scores = {key : [] for key in models}
    
    for i in range(iterations):
        if with_noise:
            duplicate_idx = np.random.choice(500, size=100, replace=False)

            noisy_features = data[:, :5] + noise_level*data[:, -5:]
            noisy_features =  np.concatenate([noisy_features, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

            noisy_scores = np.concatenate([data[:, 6], data[duplicate_idx, 6]])
            noisy_labels = np.concatenate([data[:, 5], data[duplicate_idx, 5]])

            X_train, X_test, y_train, y_test = train_test_split(
                noisy_features, 
                noisy_scores, 
                test_size=.3,
                #random_state=42,
                stratify=noisy_labels,
                shuffle=True
            )
                    
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[:, 0:5], 
                data[:, 6], # score column of dataset
                test_size=.3,
                #random_state=42,
                stratify=data[:, 5],
                shuffle=True
            )

        rn_model_idx = np.argsort(y_test)[:500]#MinMaxScaler().fit_transform(abs(y_test).reshape(-1,1)) #MinMaxScaler().fit_transform([y_test)

        for name, clf in zip(models.keys(), models.values()):
            
            clf.fit(X_train)

            y_test_scores = clf.decision_function(X_test)  # outlier scoress
            

            err_score = len(np.intersect1d(rn_model_idx, np.argsort(y_test_scores)[-500::])) / 500 # Overlap of the top 500 outliers.

            rank_scores[name].append(err_score)

    return rank_scores

def algo_rank_local_results(data, label, iterations = 10, with_noise=True, noise_level = 0, silent = False):
    rank_scores = {key : [] for key in models}
    '''if anom_type == "local":
        idx = -1
    else:'''
    idx = -1

    for i in range(iterations):
        if with_noise:
            duplicate_idx = np.random.choice(500, size=100, replace=False)

            noisy_features = data[:, :5] + noise_level*data[:, -6:-1]
            noisy_features =  np.concatenate([noisy_features, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

            noisy_scores = np.concatenate([data[:, idx], data[duplicate_idx, idx]])
            noisy_labels = np.concatenate([data[:, 5], data[duplicate_idx, 5]])

            # Should I make use of training score?
            X_train, X_test, y_train, y_test = train_test_split(
                noisy_features, 
                noisy_scores, 
                test_size=.3,
                #random_state=42,
                stratify=noisy_labels,
                shuffle=True
            )
                    
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[:, 0:5], 
                data[:, idx], # score column of dataset
                test_size=.3,
                #random_state=42,
                stratify=data[:, 5],
                shuffle=True
            )

        rn_model_idx = np.argsort(y_test)[-500::]

        for name, clf in zip(models.keys(), models.values()):
            
            clf.fit(X_train)
            # get the prediction labels and outlier scores of the training data
            #y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            #y_train_scores = clf.decision_scores_  # raw outlier scores
            

            # get the prediction on the test data
            #y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
            y_test_scores = clf.decision_function(X_test)  # outlier scoress
            
            '''rn_algo_scores = y_test_scores - y_test_scores.min()
            rn_algo_scores /= rn_algo_scores.max()
            
            rn_model_scores = abs(y_test - y_test.max())
            rn_model_scores /= rn_model_scores.max()
            '''
            #acc = balanced_accuracy_score(y_test, label*y_test_pred)
            #model_scores = anom_gen.gmm.score_samples(X_test)
            ''' print(np.argsort(y_test_scores)[::-1])
                print(np.argsort(y_test))'''
            #print(y_test_scores, model_scores.min())


            err_score = len(np.intersect1d(rn_model_idx, np.argsort(y_test_scores)[-500::])) / 500 # Overlap of the top 100 outliers.
            rank_scores[name].append(err_score)
            
            #rank_scores[name].append(kendalltau(rn_model_scores, rn_algo_scores).correlation)
            #print(f"Name: {name}, spearman: {spearmanr(rn_model_scores, rn_algo_scores)}")
            '''if not silent:
                print(name)
                print(f"Accuracy metric: {rank_scores[name]})")
                print(("------"))'''

    return rank_scores


def result_df(scores, anom_type):
    _result_df = pd.DataFrame(
        columns = ["Algorithm", "score", "type", "metric"]
    )

    for model in scores:
        for metric in scores[model]:
            for score in scores[model][metric]:
                _result_df.loc[len(_result_df)] = [model, score, anom_type, metric]
    return _result_df

def plot_results(scores, anom_type, title):
    _result_df = pd.DataFrame(
        columns = ["Algorithm", "score", "type", "metric"]
    )

    for model in scores:
        for metric in model:
            for score in scores[model][metric]:
                _result_df.loc[len(_result_df)] = [model, score, anom_type, metric]
            
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xticklabels(labels = models.keys(), rotation=45)
    ax.set_title(title)
    ax.set_ylabel("Accuracy score")
    sns.boxplot(
        data=_result_df,
        x="Algorithm",
        y="score",
        ax = ax,
    )
    return fig, ax

def al_results(data, label, iterations = 10, with_noise=True, noise_level = .25, silent = False):

    label_counts = []
    f1_scores = []
    acc_scores = []
    
    normal, data = np.split(data, [20000])

    for i in np.arange(1, 40, .5):
    
        label_scoresf1 = []
        label_scoresba = []
            
        for _ in range(iterations):
            if with_noise:
                #duplicate_idx = np.random.choice(500, size=100, replace=False)

                noisy_features = data[:, :5] + data[:, -5:] # 3 sigma noise
                
                noisy_features =  np.concatenate([noisy_features]) #, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

                noisy_labels = np.concatenate([data[:, 5]]) #, data[duplicate_idx, 5]])
                
                # Should I make use of training score?
                X_train, X_test, y_train, y_test = train_test_split(
                    noisy_features, 
                    noisy_labels, # score column of dataset
                    test_size=.01*i,
                    stratify=noisy_labels,
                    shuffle=True,
                    #random_seed=42 # Is this a good way to regualarise the peakiness.
                )

                y_train = np.array(y_train/float(label), dtype="int64")
                y_test = np.array(y_test/float(label), dtype="int64")

                X_test = np.concatenate([normal[:, 0:5], X_test])
                y_test = np.concatenate([np.zeros(len(normal[:, 0:5])), y_test])

                
                
                bst = xgb.XGBClassifier(max_depth = 25).fit(X_test, y_test)
                #print(len(X_test), y_test.sum())

                y_test_pred = bst.predict(X_train)#rfc.predict(X_train)
                label_scoresf1.append(f1_score(y_train, y_test_pred))
                label_scoresba.append(balanced_accuracy_score(y_train, y_test_pred))

        f1_scores.append(label_scoresf1)
        acc_scores.append(label_scoresba)
        label_counts.append(y_test.sum())
        
    return label_counts, acc_scores, f1_scores, 

def algo_rank_local_results(data, label, iterations = 10, with_noise=True, noise_level = 0, silent = False):
    rank_scores = {key : [] for key in models}
    '''if anom_type == "local":
        idx = -1
    else:'''
    idx = -1

    for i in range(iterations):
        if with_noise:
            duplicate_idx = np.random.choice(500, size=100, replace=False)

            noisy_features = data[:, :5] + noise_level*data[:, -6:-1]
            noisy_features =  np.concatenate([noisy_features, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

            noisy_scores = np.concatenate([data[:, idx], data[duplicate_idx, idx]])
            noisy_labels = np.concatenate([data[:, 5], data[duplicate_idx, 5]])

            # Should I make use of training score?
            X_train, X_test, y_train, y_test = train_test_split(
                noisy_features, 
                noisy_scores, 
                test_size=.3,
                #random_state=42,
                stratify=noisy_labels,
                shuffle=True
            )
                    
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[:, 0:5], 
                data[:, idx], # score column of dataset
                test_size=.3,
                #random_state=42,
                stratify=data[:, 5],
                shuffle=True
            )

        rn_model_idx = np.argsort(y_test)[-500::]

        for name, clf in zip(models.keys(), models.values()):
            
            clf.fit(X_train)
 
            y_test_scores = clf.decision_function(X_test)  # outlier scoress

            err_score = len(np.intersect1d(rn_model_idx, np.argsort(y_test_scores)[-500::])) / 500 # Overlap of the top 100 outliers.
            rank_scores[name].append(err_score)

    return rank_scores

'''def al_results(data, label, iterations = 10, with_noise=True, noise_level = .1, silent = False):
    label_counts = []
    f1_scores = []
    acc_scores = []
    for i in np.arange(1, 20, .5):
    
        label_scoresf1 = []
        label_scoresba = []

        for _ in range(iterations):
            if with_noise:
                #duplicate_idx = np.random.choice(500, size=100, replace=False)

                noisy_features = data[:, :5] + data[:, -5:] # 3 sigma noise
                noisy_features =  np.concatenate([noisy_features]) #, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

                noisy_labels = np.concatenate([data[:, 5]]) #, data[duplicate_idx, 5]])
                
                # Should I make use of training score?
                X_train, X_test, y_train, y_test = train_test_split(
                    noisy_features, 
                    noisy_labels, # score column of dataset
                    test_size=.01*i,
                    stratify=noisy_labels,
                    shuffle=True,
                    #random_seed=42 # Is this a good way to regualarise the peakiness.
                )

                y_train = np.array(y_train/float(label), dtype="int64")
                y_test = np.array(y_test/float(label), dtype="int64")

                bst = xgb.XGBClassifier(max_depth = 10).fit(X_test, y_test)

                y_test_pred = bst.predict(X_train)
                label_scoresf1.append(f1_score(y_train, y_test_pred))
                label_scoresba.append(balanced_accuracy_score(y_train, y_test_pred))

        f1_scores.append(np.mean(label_scoresf1))
        acc_scores.append(np.mean(label_scoresba))
        label_counts.append(y_test.sum())
 
    return label_counts, acc_scores, f1_scores
'''