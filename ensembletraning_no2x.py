import os
import shutil
import re
from ensphymodel2train  import BtEnsPDEModelPols
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Set the root directory for model output
rootpath = '/devb/py2testing/demo_test/test_phymodel'

# Create the root directory if it doesn't exist
if not os.path.exists(rootpath):
    # Note: shutil.rmtree line is commented out, would remove directory if uncommented
    os.makedirs(rootpath)

# Initialize the NOx ensemble PDE model with specified parameters
# This appears to be a custom model class for pollution modeling
mynoxensModel = BtEnsPDEModelPols(th_low=None, th_high=None, logno2max=5.5, lognoxmax=6.5, rootpath=rootpath)

# Load the data for the model
mynoxensModel.loadData()

# Define the covariates (predictor variables) for the model
# These include temporal variables (iweek, month, year),
# spatial variables (lat, lon, elevation),
# traffic emissions (various AADT variables),
# land use features (impervious surfaces, NDVI),
# meteorological variables (temperature, wind speed),
# air pollutant concentrations and other environmental factors
tcovs = ['iweek', 'lat', 'lon', 'ele30m', 'x2', 'y2', 'xy',
         'int_d_300m_c1', 'int_d_300m_c2', 'int_d_300m_c3', 'int_d_300m_c4',
         'int_d_5km_c1', 'int_d_5km_c2', 'int_d_5km_c3', 'int_d_5km_c4',
         'NONFWY_Light_AADT_50res_5000neigh_emissions', 'NONFWY_Light_AADT_50res_300neigh_emissions',
         'NONFWY_Heavy_AADT_50res_5000neigh_emissions', 'NONFWY_Heavy_AADT_50res_300neigh_emissions',
         'FWY_Light_AADT_50res_5000neigh_emissions', 'FWY_Light_AADT_50res_300neigh_emissions',
         'FWY_Heavy_AADT_50res_5000neigh_emissions', 'FWY_Heavy_AADT_50res_300neigh_emissions',
         'airport_distance', 'airport_highdistance', 'impervious', 'impervious_15km', 'rmax', 'rmin',
         'sph', 'srad', 'tmmn', 'tmmx', 'vs', 'BCMASS_BOT', 'CO_BOT', 'DD_O3_BOT', 'GMITO3',
         'NIMASS25_BOT', 'NO_BOT', 'NO2_BOT', 'O3_BOT', 'PL_BOT', 'PS', 'SSMASS25_BOT', 'NDVI1km',
         'NDVI15km', 'ozone24mean', 'Population_300m', 'ele30m', 'poirest', 'poifast', 'no2_pv1',
         'no2_pv2', 'nox_pv1', 'nox_pv2', 'U2M', 'V2M', 'U10M', 'V10M', 'U50M', 'V50M', 'PBLH', 'stagnation',
         'mixing', 'day_y', 'month', 'year', 'day_y_sine', 'day_y_cos', 'r50nlcd_22_1km_log',
         'r50nlcd_23_1km_log', 'r50nlcd_24_1km_log', 'r50nlcd_21_1km_log', 'r50nlcd_31_1km_log',
         'r50nlcd_22_15km_log', 'r50nlcd_23_15km_log', 'r50nlcd_24_15km_log',
         'r50nlcd_21_15km_log', 'r50nlcd_31_15km_log']

# Preprocess the data using the defined covariates
mynoxensModel.preprocessing(tcovs)

# Define which models to train (model 1 and 2)
unmodels = [1, 2]
for i in unmodels:
    # Set up paths for the current model
    modelPath = rootpath + '/model_' + str(i)
    samplepath = rootpath + '/model_' + str(i) + '/sample'

    # Create or recreate the model directory
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    else:
        # If directory exists, remove it and create fresh
        shutil.rmtree(modelPath)
        os.makedirs(modelPath)

    print("Training ", modelPath, " ... ...")

    # Create the sample directory for storing sampling results
    if not os.path.exists(samplepath):
        os.makedirs(samplepath)

    # Create the path for saving the trained model
    savepath = modelPath + '/model'
    os.makedirs(savepath)

    # Train the model with specified parameters
    # This trains an ensemble model for NOx prediction
    # with 100 epochs, saving samples and the final model
    mynoxensModel.ensAModel('newid2', 'source', 'month_date',
                            nepoch=100, samplepath=samplepath, savePath=savepath)
