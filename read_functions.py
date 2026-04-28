import rioxarray
from pathlib import Path

###############################################
##### open_to_pd_df_withregionlabel () #######
###############################################

def open_to_pd_df_withregionlabels(path):

    da = rioxarray.open_rasterio(path)
    if 'long_name' in da.attrs:
        da = da.assign_coords(band=list(da.attrs['long_name']))

    raster = da.drop_sel(band="std")  # not useful and NA most of the time
    df = raster.stack(samples=("y", "x")).to_pandas().T.dropna().reset_index()

    df["region"] = Path(path).stem
    df["trainclass"] = (df["trainclass"] * 10000).round().astype(int)
    df["region_class"] = (df["region"].astype(str) + "_" + df['trainclass'].astype(str))
    return df

###############################################
############ open_to_pd_df () ################
###############################################

# combining to_clean_pd_df() and open_with_labels()

def open_to_pd_df(path):

    da = rioxarray.open_rasterio(path)
    if 'long_name' in da.attrs:
        da = da.assign_coords(band=list(da.attrs['long_name']))

    raster = da.drop_sel(band="std")  # not useful and NA most of the time
    df = raster.stack(samples=("y", "x")).to_pandas().T.dropna().reset_index()
    df["trainclass"] = (df["trainclass"] * 10000).round().astype(int)
    return df


###############################################
############ to_clean_pd_df () ################
###############################################

def to_clean_pd_df(x):

    raster = x.drop_sel(band="std")  # not useful and NA most of the time
    df = raster.stack(samples=("y", "x")).to_pandas().T.dropna().reset_index()
    return df

###############################################
############ open_with_labels()################
###############################################


def open_with_labels(path): # small helper function, so the band labels dont get deleted during rioxarray.open_rasterio()
    da = rioxarray.open_rasterio(path)
    if 'long_name' in da.attrs:
        da = da.assign_coords(band=list(da.attrs['long_name']))
    return da
