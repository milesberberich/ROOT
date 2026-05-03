import pandas as pd
import tifffile as tiff
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

###############################################
############## map_prediction() ##############
###############################################

def map_predictions(all_coordinates, cv_predictions, region_series, templates, output_dir):
    """
    Consolidated function to filter results by region and save multiple aligned TIFFs
    with the band named 'trainclass'.
    """

    df_final = pd.DataFrame(all_coordinates, columns=['x', 'y'])
    df_final['pred'] = cv_predictions
    df_final['region'] = region_series.values

    for region_name, template_path in templates.items():
        region_results = df_final[df_final['region'] == region_name]

        if region_results.empty:
            print(f"Skipping {region_name}: No results found for this region.")
            continue

        with rasterio.open(template_path) as src:
            profile = src.profile
            transform = src.transform
            width, height = src.width, src.height

        profile.update(dtype='float32', count=1, nodata=0, compress='lzw')
        pixel_matrix = np.zeros((height, width), dtype='float32')
        rows, cols = rowcol(transform, region_results['x'].values, region_results['y'].values)

        preds = region_results['pred'].values
        for r, c, p in zip(rows, cols, preds):
            if 0 <= r < height and 0 <= c < width:
                pixel_matrix[r, c] = p

        out_path = Path(output_dir) / f"prediction_{region_name}.tif"
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pixel_matrix, 1)

            dst.set_band_description(1, 'trainclass')

        print(f"Successfully saved aligned raster for {region_name} at {out_path}")

###############################################
############## rf_sample() ####################
###############################################

def rf_sample(df, method = None, total_n = 50000, weight = None, oversampling_number = None):

    ###########################
    if method == "weighted":

        target_class = 2

        prop = (df['trainclass'] == target_class).mean()
        n_target = int(total_n * prop * weight)
        n_target = min(n_target, (df['trainclass'] == target_class).sum())

        # Sample both groups and combine
        target_df = df[df['trainclass'] == target_class].sample(n_target)
        others_df = df[df['trainclass'] != target_class].sample(total_n - n_target)

        return pd.concat([target_df, others_df]).sample(frac=1)

    ###########################
    if method == "undersampling":

        min_size = df["trainclass"].value_counts().min()
        balanced_list = []
        print("Through undersampling, the ration of regions can change!")
        for class_id in df["trainclass"].unique():
            class_subset = df[df["trainclass"] == class_id]
            balanced_list.append(class_subset.sample(n=min_size, random_state=42))

        return pd.concat(balanced_list).reset_index(drop=True)


    ###########################
    if method == "oversampling":

        df_dead = df[df["trainclass"] == 2]
        df_dead_os = df_dead.sample(oversampling_number, replace=True, random_state=42)
        print("Through oversampling, the ration of regions can change!")
        balanced_list = [df_dead_os]
        for class_id in [1, 3]:
            class_subset = df[df["trainclass"] == class_id]
            replace_needed = len(class_subset) < oversampling_number
            balanced_list.append(class_subset.sample(n=oversampling_number, replace=replace_needed, random_state=42))

        return pd.concat(balanced_list).reset_index(drop=True)

    else:
        print("Please specify method.")


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
