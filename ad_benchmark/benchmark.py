from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.cd import CD
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.loci import LOCI
from pyod.models.pca import PCA


def init_algos(contamination):
    return {
        "KNN": KNN(contamination = contamination),
        "IForest": IForest(contamination = contamination),
        #"ECOD": ECOD(),
        "LOF": LOF(contamination = contamination),
        "ABOD": ABOD(contamination = contamination),
        "CBLOF": CBLOF(contamination = contamination),
        "CD": CD(contamination = contamination),
        "COPOD": COPOD(contamination = contamination),
        "Feature Bagging": FeatureBagging(contamination = contamination),
        "HBOS": HBOS(contamination = contamination)
    }
    
models = {
    "KNN": KNN(),
    "IForest": IForest(),
    #"ECOD": ECOD(),
    "LOF": LOF(),
    "ABOD": ABOD(),
    "CBLOF": CBLOF(),
    #"COF": COF(),
    "CD": CD(),
    "COPOD": COPOD(),
    "Feature Bagging": FeatureBagging(),
    "HBOS": HBOS(),
    #"INNE": INNE(),
    #"KDE": KDE(),
    #"LOCI": LOCI(),
    #"PCA": PCA(),
} 

model_outputs = {
}
for model in models:
    model_outputs[model] = {
        "local": [],
        "global": [],
        "cluster": [],
        "combined": []
    }

anom_types = {
    "local" : [], 
    "global" : [],
    "cluster" : [],
    "combined" : []
}