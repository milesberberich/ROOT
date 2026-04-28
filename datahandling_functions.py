import pandas as pd
###############################################
############ get_weighted_sample()################
###############################################


def get_weighted_sample(df, total_n=30000, target_class=2, multiplier=2): # used to increase the performance on the minority class in the unbalanced data set. The model gets trained on more deadwood pixels.

    prop = (df['trainclass'] == target_class).mean()
    n_target = int(total_n * prop * multiplier)
    n_target = min(n_target, (df['trainclass'] == target_class).sum())

    # Sample both groups and combine
    target_df = df[df['trainclass'] == target_class].sample(n_target)
    others_df = df[df['trainclass'] != target_class].sample(total_n - n_target)

    return pd.concat([target_df, others_df]).sample(frac=1)



###############################################
############ balance_dataset() ################
###############################################



def balance_dataset(df_input, mode, target_number):
    """
    Balances a dataframe based on the BALANCE_MODE settings.
    """
    if mode == "OVERSAMPLING":
        # Class 2 is Deadwood
        df_dead = df_input[df_input["trainclass"] == 2]

        # Sample with replacement to reach target_number
        df_dead_os = df_dead.sample(target_number, replace=True, random_state=42)

        balanced_list = [df_dead_os]
        # Class 1 = clear, 3 = undisturbed
        for class_id in [1, 3]:
            class_subset = df_input[df_input["trainclass"] == class_id]
            # Sample to target_number (using replacement if subset is smaller than target)
            replace_needed = len(class_subset) < target_number
            balanced_list.append(class_subset.sample(n=target_number, replace=replace_needed, random_state=42))

        return pd.concat(balanced_list).reset_index(drop=True)

    elif mode == "UNDERSAMPLING":
        # Find the smallest class in the specific input dataframe
        min_size = df_input["trainclass"].value_counts().min()
        balanced_list = []
        for class_id in df_input["trainclass"].unique():
            class_subset = df_input[df_input["trainclass"] == class_id]
            balanced_list.append(class_subset.sample(n=min_size, random_state=42))
        return pd.concat(balanced_list).reset_index(drop=True)

    return df_input
