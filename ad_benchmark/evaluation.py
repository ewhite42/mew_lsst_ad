from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

def algo_results(data, models, label, iterations = 10, with_noise=True, noise_level = 1, silent = False):

    ## initialize a nested dictionary where the keys correspond
    ## to the keys from the models dictionary (and thus the names of 
    ## each unsupervised anomaly detection method), and each nested dictionary
    ## has keys "BAL_ACC" and "F1" associated with empty lists
    scores = {key : {
        "BAL_ACC": [],
        "F1": []
    } for key in models}

    for name, clf in zip(models.keys(), models.values()):
            
        for i in range(iterations):
            if with_noise:
                #duplicate_idx = np.random.choice(500, size=100, replace=False)
                
                ## why does it select the last five columns when the last two columns are tsne data? -MEW
                noisy_features = data[:, :5] + data[:, -5:] # 3 sigma noise
                noisy_features =  np.concatenate([noisy_features]) #, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

                noisy_labels = np.concatenate([data[:, 5]]) #, data[duplicate_idx, 5]])
                
                # Should I make use of training score?
                
                ## split the data into random x,y training and test datasets -MEW
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

            ## fit the model that's selected in the loop to the X training set -MEW
            clf.fit(X_train)

            ## get the prediction from the test data (returns outlier labels, 0 or 1) -MEW, from PyOD docs
            y_test_pred = clf.predict(X_test)  
            #y_test_scores = clf.decision_function(X_test) 
            
            ## get the F1 score for this model as applied to this dataset: -MEW
            f1 = f1_score(np.array(y_test)/label, y_test_pred)
            
            ## get the balanced accuracy score for this model as applied to this dataset: -MEW
            balanced_accuracy =  balanced_accuracy_score(y_test, label*y_test_pred)
            
            ## fill in the F1 and balanced accuracy scores for each model 
            ## in the empty lists in the scores dictionary initialized above.
            ## Note that a value for F1 and BAL_ACC are returned for each of the 
            ## 10 iterations through the loop -MEW
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

        rn_model_idx = np.argsort(y_test)[:500] #MinMaxScaler().fit_transform(abs(y_test).reshape(-1,1)) #MinMaxScaler().fit_transform([y_test)

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

    ## initialize Pandas dataframe to store results -MEW
    _result_df = pd.DataFrame(
        columns = ["Algorithm", "score", "type", "metric"]
    )

    ## loop through each model, then through both F1 and BAL_ACC, then through
    ## each of the scores for each metric -MEW
    for model in scores:
        for metric in scores[model]:
            for score in scores[model][metric]:
                ## append a row to _result_df with the model, score, anomaly type, and metric -MEW
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
    
    ## split the data into two parts -- the first part consists of the first 20,000 samples (normal),
    ## the second part consists of the last samples (data) -- MEW
    normal, data = np.split(data, [20000])

    for i in np.arange(1, 40, .5):
    
        label_scoresf1 = []
        label_scoresba = []
            
        for _ in range(iterations):
            if with_noise:
                #duplicate_idx = np.random.choice(500, size=100, replace=False)

                ## selects first and last five columns of each row of
                ## data, which I imagine are the orbital / color parameters 
                ## and associated noise measures - MEW
                noisy_features = data[:, :5] + data[:, -5:] # 3 sigma noise
                
                noisy_features =  np.concatenate([noisy_features]) #, noisy_features[duplicate_idx, 0:5]]) # Add duplicates

                ## create array containing the label of each datapoint (row) in data array -MEW
                noisy_labels = np.concatenate([data[:, 5]]) #, data[duplicate_idx, 5]])
                
                ## split noisy_features into test and train data sets -MEW
                # Should I make use of training score?
                X_train, X_test, y_train, y_test = train_test_split(
                    noisy_features, 
                    noisy_labels, # score column of dataset
                    test_size=.01*i,
                    stratify=noisy_labels,
                    shuffle=True,
                    #random_seed=42 # Is this a good way to regualarise the peakiness.
                )

                ## divide all y train and test data by the label number so that you
                ## can just sum up the 1's in the y test array to get the total number
                ## of labelled anomalies -MEW
                y_train = np.array(y_train/float(label), dtype="int64")
                y_test = np.array(y_test/float(label), dtype="int64")

                ## append data from the first 5 columns of the normal array to 
                ## the beginning of the X_test array, and append an array of zeros
                ## (length same as normal array) to the beginning of the y_test array -MEW
                X_test = np.concatenate([normal[:, 0:5], X_test])
                y_test = np.concatenate([np.zeros(len(normal[:, 0:5])), y_test])

                ## initialize XGBoost classifier and fit to the x and y test datasets -MEW
                bst = xgb.XGBClassifier(max_depth = 25).fit(X_test, y_test)
                #print(len(X_test), y_test.sum())

                ## use XGBoost to predict the labels for each sample in X_train (I think?) -MEW
                y_test_pred = bst.predict(X_train)#rfc.predict(X_train)
                
                ## compare true labels (y_train) with predicted labels (y_test_pred) to determine
                ## how well XGBoost was able to determine whether a datapoint was an anomaly or not;
                ## accuracies represented by the F1 and balanced accuracy scores -MEW
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
