
from sklearn.ensemble import RandomForestClassifier

###############################################
##### randomForestClass () #######
###############################################

def randomForestClass(ntrees = 750, pred_train = None, forestclass_train = None):
    """
    Handles class weights.
    """
    if FIT == "balanced":
        rf = RandomForestClassifier(n_estimators=ntrees, class_weight="balanced_subsample", random_state=42)
        print("balanced scikit learn mode was used.")

    else:
        rf = RandomForestClassifier(n_estimators=ntrees, random_state=42)
        print("unbalanced scikit learn mode was used.")

    rf.fit(pred_train, forestclass_train)
    return rf