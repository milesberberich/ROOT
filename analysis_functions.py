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

###############################################
##### statistics () #######
###############################################
def statistics(df):
    print("\n--- Dataset Statistics ---")

    summary_counts = pd.crosstab(df['region'], df['trainclass'])
    summary_pct = pd.crosstab(df['region'], df['trainclass'], normalize='all') * 100
    print("Number of total valid pixels per region and class:")
    print(summary_counts)
    print("\nTotal valid pixels per region/class (in % of total dataset):")
    print(summary_pct.round(2).astype(str) + " %")

    import rioxarray
    import numpy as np

    def predict_and_export_raster(rf_model, input_path, output_path, band_names):
        """
        Loads a multi-band TIFF, predicts classes for every pixel using a trained
        Random Forest model, and exports the result as a new TIFF.
        """
        ds = rioxarray.open_rasterio(input_path).assign_coords(band=band_names)

        raw_data = ds.sel(band=band_names).values
        n_bands, height, width = raw_data.shape
        X_flat = raw_data.reshape(n_bands, -1).T

        valid_mask = np.all(np.isfinite(X_flat), axis=1)

        # 4. Initialize the output array with a no-data value (-9999)
        predictions = np.full(X_flat.shape[0], -9999.0)

        # 5. Run prediction only on the valid pixels[cite: 1]
        if np.any(valid_mask):
            predictions[valid_mask] = rf_model.predict(X_flat[valid_mask])

        # 6. Reshape predictions back to the original 2D grid dimensions[cite: 1]
        prediction_grid = predictions.reshape(height, width)

        # 7. Create an output DataArray using the spatial metadata of the input[cite: 1]
        output_da = ds.isel(band=0).copy(data=prediction_grid)

        # 8. Export to .tif with the correct no-data value[cite: 1]
        output_da.rio.to_raster(output_path, nodata=-9999.0)

        print(f"Successfully exported classification to: {output_path}")