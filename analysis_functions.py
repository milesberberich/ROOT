from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

###############################################
##### randomForestClass () #######
###############################################

def randomForestClass(ntrees = 750, pred_train = None, forestclass_train = None, FIT = "d"):
    """
    Handles class weights.
    """
    if FIT == "balanced":
        rf = RandomForestClassifier(n_estimators=ntrees, class_weight="balanced_subsample", random_state=42)
        print("balanced scikit learn mode was used.")

    if FIT != "balanced":
        rf = RandomForestClassifier(n_estimators=ntrees, random_state=42)
        print("unbalanced scikit learn mode was used.")

    rf.fit(pred_train, forestclass_train)
    return rf

def statistics(df):
    # %%

    print("\n")
    print("\n")
    print("Total valid pixels per dataset in percent")
    summary_table = summary_table/summary_table.sum().sum()*100
    print(summary_table)

    print("\n")
    print("\n")

    print("Number of total valid pixels per dataset")
    summary_table = pd.crosstab(df['region'], df["trainclass"])
    print(summary_table)