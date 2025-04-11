seeValue = 100
import os

os.environ['PYTHONHASHSEED'] = str(seeValue)
from keras.backend import manual_variable_initialization

manual_variable_initialization(True)

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from model.metrics import r2np, rmse2np
from numpy.random import choice
from os.path import exists
from datetime import datetime
import pickle
import re
import gc

# import dill as pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model.phy2deepmodel import PhysicsResAutocoderPols
from model.baselinemodel import baselineFRNN

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class BtEnsPDEModelPols:
    """
    Ensemble learning class that calls jPINN (Physics-Informed Neural Network) or a baseline FRNN (Full Residual Neural Network)
    for modeling pollutant distributions.

    This class handles data loading, preprocessing, model training with bootstrapping, and evaluation
    of physics-constrained deep learning models for environmental pollutant concentration prediction.

    The class supports both fully supervised learning and semi-supervised approaches with a focus
    on NO2/NOx or PM2.5/PM10 prediction in environmental monitoring applications.
    """

    def __init__(self, th_low=None, th_high=None, logno2max=5.5, lognoxmax=6.5, rootpath="/tmp"):
        """
        Initialize the Ensemble PDE Model for pollutant prediction.

        Parameters:
        -----------
        th_low : float, optional
            Lower threshold for oversampling strategies (default None).
        th_high : float, optional
            Upper threshold for oversampling strategies (default None).
        logno2max : float, optional
            Maximum log-transformed NO2 value for physics constraints (default 5.5).
        lognoxmax : float, optional
            Maximum log-transformed NOx value for physics constraints (default 6.5).
        rootpath : str, optional
            Root directory for storing model outputs and intermediate files (default "/tmp").
        """
        self.__version__ = '0.0.1'
        self.th_low = th_low
        self.th_high = th_high
        self.logno2max = logno2max  # Max log-transformed NO2 value for physics constraints
        self.lognoxmax = lognoxmax  # Max log-transformed NOx value for physics constraints
        self.rootpath = rootpath  # Base path for storing model outputs

    def loadData(self, tfl=None):
        """
        Load and preprocess the environmental monitoring data.

        This method reads a CSV file containing pollutant measurements and site information,
        performs initial data cleaning (handling zeros, log transformations, feature engineering),
        and splits data into full samples (with both NO2 and NOx) and semi-supervised samples.

        Parameters:
        -----------
        tfl : str, optional
            Path to the input CSV file. If None, uses a default path.

        Notes:
        ------
        - Transforms zero values in pollutant measurements to small non-zero values
        - Creates log-transformed variables and engineered features
        - Parses dates and creates temporal variables
        - Separates data into fully labeled and semi-supervised sets
        """
        if tfl is None:
            # Default data file path if none provided
            tfl = "/deva/phy2data/merged_fdata_uploc_first.csv"
        print(tfl)

        # Load dataset from CSV
        data = pd.read_csv(tfl)
        data['newid2'] = data['newid2'].astype(str)

        # Handle zero values for log transformation (replace with small values)
        sindex = data[data['no2'] == 0].index
        data.loc[sindex, 'no2'] = 0.003  # Replace 0 with small value for NO2
        sindex = data[data['nox'] == 0].index
        data.loc[sindex, 'nox'] = 0.05  # Replace 0 with small value for NOx

        # Create combined month-date feature
        data['month_date'] = data.apply(lambda x: str(x['month']) + '.' + x['start_date'], axis=1)

        # Log transform pollutant measurements
        data['no2_log'] = np.log(data['no2'])
        data['nox_log'] = np.log(data['nox'])

        # Create polynomial terms for spatial coordinates
        data['x2'] = pow(data['x'], 2)
        data['y2'] = pow(data['y'], 2)
        data['xy'] = data['x'] * data['y']

        # Parse date strings into datetime objects
        data['start_date1'] = data['start_date'].apply(datetime.strptime, args=('%Y-%m-%d',))

        # Log transform land cover variables with small offset to avoid log(0)
        # NLCD codes: 21=developed open space, 22=developed low intensity, 23=developed medium intensity,
        # 24=developed high intensity, 31=barren land
        land_cover_vars = [
            'r50nlcd_22_1km', 'r50nlcd_23_1km', 'r50nlcd_24_1km', 'r50nlcd_21_1km', 'r50nlcd_31_1km',
            'r50nlcd_22_15km', 'r50nlcd_23_15km', 'r50nlcd_24_15km', 'r50nlcd_21_15km', 'r50nlcd_31_15km'
        ]
        for var in land_cover_vars:
            data[f'{var}_log'] = np.log(data[var] + 0.00001)

        print("Initial data sample size:", data.shape[0])
        print('data size:', data.shape)

        # Identify semi-supervised data (sites with suffix _c{digit} lack NO2/NOx measurements)
        selindex = data[data['newid2'].str.contains(re.compile('_c[0-9]{1}$'), na=False)].index
        data.loc[selindex, 'no2'] = np.nan
        data.loc[selindex, 'nox'] = np.nan
        self.semidata = data.loc[selindex]  # Data without pollutant measurements

        # Select fully labeled data where both NO2 and NOx are available
        selindex = data[((data['no2'].notna()) & (data['nox'].notna()))].index
        self.fulldata = data.loc[selindex]  # Data with pollutant measurements

        print('fulldata:', self.fulldata.shape, '; semidata:', self.semidata.shape)
        return

    def preprocessing(self, tcovs, outtype=2):
        """
        Standardize features and target variables for model training.

        This method applies standard scaling (zero mean, unit variance) to both
        predictor variables and target variables, and saves the scalers for later use.

        Parameters:
        -----------
        tcovs : list
            List of column names to use as predictor variables.
        outtype : int, optional
            Output variable selection: 0=NO2 only, 1=NOx only, 2=both NO2 and NOx (default 2).

        Notes:
        ------
        - Drops rows with missing values in any predictor variable
        - Standardizes both inputs and outputs
        - Saves the data and scalers to disk for later use
        """
        # Remove rows with missing predictors
        self.fulldata = self.fulldata.dropna(subset=tcovs)

        # Save cleaned data to CSV
        dataFl = self.rootpath + '/cleanfullDataup.csv'
        self.fulldata.to_csv(dataFl, index=False)

        # Extract and standardize predictors
        X = self.fulldata[tcovs].values
        self.scX = preprocessing.StandardScaler().fit(X)
        self.Xn = self.scX.transform(X)
        print('tcovs used for training:', tcovs)

        # Extract and standardize target variables based on outtype
        if outtype == 2:
            y = self.fulldata[['no2_log', 'nox_log']].values  # Both NO2 and NOx
        elif outtype == 0:
            y = self.fulldata[['no2_log']].values  # NO2 only
        elif outtype == 1:
            y = self.fulldata[['nox_log']].values  # NOx only

        self.scy = preprocessing.StandardScaler().fit(y)
        self.yn = self.scy.transform(y)

        # Save scalers to disk using pickle
        normPath = self.rootpath + '/scX_scalor.pkl'
        with open(normPath, 'wb') as fl:
            pickle.dump(self.scX, fl)

        normPath = self.rootpath + '/scy_scalor.pkl'
        with open(normPath, 'wb') as fl:
            pickle.dump(self.scy, fl)

    def oversampling(self, sindex, target, th_low=None, th_high=None, samprop=0.2):
        """
        Perform selective oversampling to balance the dataset for extreme values.

        This method identifies samples with very high or very low pollutant values
        and adds duplicate samples to improve model performance on these extremes.

        Parameters:
        -----------
        sindex : numpy.ndarray
            Indices of samples to potentially oversample from.
        target : str
            Target variable name for selective oversampling (e.g., 'nox_m').
        th_low : float, optional
            Threshold below which samples are considered low extreme values (default None).
        th_high : float, optional
            Threshold above which samples are considered high extreme values (default None).
        samprop : float, optional
            Proportion of extreme samples to add (default 0.2).

        Returns:
        --------
        numpy.ndarray
            Updated indices after oversampling.

        Notes:
        ------
        - If both th_low and th_high are None, returns original indices
        - Otherwise adds selected extreme value samples to balance distribution
        """
        # Identify and oversample low extreme values
        if th_low is not None:
            smallindex = np.where(self.fulldata[target] <= th_low)
            mask = np.isin(smallindex, sindex)
            trIndex_over = smallindex[0][mask[0, :]]
            n = trIndex_over.shape[0]
            smallIndex_over_r1 = np.random.choice(trIndex_over, int(n * samprop))

        # Identify and oversample high extreme values
        if th_high is not None:
            bigindex = np.where(self.fulldata[target] >= th_high)
            mask = np.isin(bigindex, sindex)
            trIndex_over = bigindex[0][mask[0, :]]
            n = trIndex_over.shape[0]
            bigIndex_over_r1 = np.random.choice(trIndex_over, int(n * samprop))

        # Return appropriate indices based on which thresholds were specified
        if th_low is None and th_high is None:
            return sindex
        if th_low is None and th_high is not None:
            sindex_over = np.concatenate((sindex, bigIndex_over_r1))
            print(sindex_over.shape, sindex.shape)
            return sindex_over
        if th_low is not None and th_high is None:
            sindex_over = np.concatenate((sindex, smallIndex_over_r1))
            print(sindex_over.shape, sindex.shape)
            return sindex_over

        # Both thresholds specified, add both low and high extreme samples
        sindex_over = np.concatenate((sindex, smallIndex_over_r1, bigIndex_over_r1))
        print(sindex_over.shape, sindex.shape)
        return sindex_over

    def bootstrapSiteSrc(self, srcStr, srcs, idStr):
        """
        Sample monitoring sites using bootstrap approach based on source type.

        This method performs weighted sampling of monitoring sites to create
        a representative independent test set, with sampling probability
        proportional to the number of samples per site.

        Parameters:
        -----------
        srcStr : str
            Column name for data source identifier.
        srcs : list
            List of source types to include (e.g., ['AQS']).
        idStr : str
            Column name for site identifier.

        Returns:
        --------
        numpy.ndarray
            Array of selected site IDs for independent testing.

        Notes:
        ------
        - Selects approximately 18% of sites
        - Sites with more samples have higher probability of selection
        - Used to create spatially independent validation sets
        """
        # Filter data by source type
        subdata = self.fulldata[self.fulldata[srcStr].isin(srcs)]

        # Count samples per site
        sgrp = subdata[idStr].value_counts()
        sitesCnt = pd.DataFrame({idStr: sgrp.index, 'samplesize': sgrp.values}, index=sgrp.index)

        # Calculate sampling probability proportional to sample size
        np.random.seed()
        cnts = sitesCnt['samplesize'].astype(np.float64)
        sitesCnt['prop'] = cnts / np.sum(cnts)

        # Select ~18% of sites with probability proportional to sample count
        tarsitesNum = int(sitesCnt.shape[0] * 0.18)
        testsiteIndex = choice(np.array([i for i in range(len(sitesCnt))]), tarsitesNum, replace=False,
                               p=sitesCnt['prop'].values)
        selsites = sitesCnt.iloc[testsiteIndex][idStr].copy()

        # Clean up memory
        del sitesCnt
        gc.collect()

        return selsites.values

    def bootstrapSelectSites(self, idStr='id', tpath=None):
        """
        Select or load previously selected sites for independent testing.

        This method either loads preselected test sites or creates and saves
        a new selection using the bootstrapSiteSrc method.

        Parameters:
        -----------
        idStr : str, optional
            Column name for site identifier (default 'id').
        tpath : str, optional
            Path to save/load site selection (default None).

        Returns:
        --------
        numpy.ndarray
            Array of selected site IDs for independent testing.

        Notes:
        ------
        - Uses a consistent set of test sites across runs if available
        - Creates new test site selection if no saved selection exists
        - Currently focused on AQS monitoring network sites
        """
        # Determine file path for saved site selection
        if tpath is None:
            pFile = '/meteodata/phase2modeltestcv/selsites.pkl'
        else:
            pFile = tpath + '/selsites.pkl'

        # Generate new site selection if no saved selection exists
        if not exists(pFile):
            btselectedSites = self.bootstrapSiteSrc('source', ['AQS'], 'newid2')
            print('AQS:', len(btselectedSites))
            with open(pFile, 'wb') as fp:  # Pickling
                pickle.dump(btselectedSites, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # Load existing site selection
        else:
            with open(pFile, 'rb') as fp:  # Unpickling
                btselectedSites = pickle.load(fp)
                btselectedSites = btselectedSites.astype('str')

        return btselectedSites

    def train_test_splitR(self, targetindex, stratified, test_size=0.2):
        """
        Split data into training and testing sets with special handling for rare strata.

        This method performs stratified sampling but ensures that rare categories
        (with fewer than 5 samples) are all included in the training set to avoid
        test sets with too few samples per stratum.

        Parameters:
        -----------
        targetindex : numpy.ndarray
            Indices of samples to split.
        stratified : str
            Column name to use for stratification.
        test_size : float, optional
            Proportion of data to use for testing (default 0.2).

        Returns:
        --------
        tuple
            (train_indices, test_indices)

        Notes:
        ------
        - Identifies strata with fewer than 5 samples
        - Places all samples from rare strata in training set
        - Performs stratified split on remaining samples
        """
        # Get actual data indices from targetindex
        dfindex = self.fulldata.iloc[targetindex].index

        # Count number of samples per stratum
        sgrp = self.fulldata.iloc[targetindex][stratified].value_counts()
        self.fulldata.loc[dfindex, 'stratifed_cnt'] = sgrp.loc[self.fulldata.loc[dfindex, stratified]].values

        # Identify indices with rare categories (<5 samples)
        pos1_index = np.where(self.fulldata['stratifed_cnt'] < 5)[0]
        # Identify indices with sufficient samples for stratified splitting
        posT_index = np.where(self.fulldata['stratifed_cnt'] >= 5)[0]

        # Perform stratified split on sufficiently represented categories
        np.random.seed()
        trainsiteIndex, testsiteIndex = train_test_split(posT_index,
                                                         stratify=self.fulldata.iloc[posT_index][stratified],
                                                         test_size=test_size)

        # Add rare categories to training set
        trainsiteIndex = np.concatenate((trainsiteIndex, pos1_index))
        print('sampling split: ', len(trainsiteIndex), len(testsiteIndex))

        return (trainsiteIndex, testsiteIndex)

    def saveModel(self, model, flmajor):
        """
        Save a Keras model in multiple formats.

        This method saves the model architecture as JSON, weights as HDF5,
        and the complete model in HDF5 format.

        Parameters:
        -----------
        model : keras.Model
            The model to save.
        flmajor : str
            Base file path/prefix for saved model files.

        Notes:
        ------
        - Saves model architecture as {flmajor}_frm.json
        - Saves model weights as {flmajor}_wei.h5
        - Saves complete model as {flmajor}_wholemodel.h5
        """
        # Serialize model architecture to JSON
        model_json = model.to_json()
        tfl = flmajor + '_frm.json'
        with open(tfl, "w") as json_file:
            json_file.write(model_json)

        # Serialize model weights to HDF5
        tfl = flmajor + '_wei.h5'
        model.save_weights(tfl)

        # Save complete model to HDF5
        tfl = flmajor + '_wholemodel.h5'
        model.save(tfl)

    def ensAModel(self, idStr='id', sitestratified='source', stratified='month_date', nepoch=2, samplepath='tmp',
                  savePath='/tmp'):
        """
        Train a Physics-Informed Neural Network (jPINN) ensemble model.

        This method handles the entire training pipeline for the physics-informed model:
        data splitting, oversampling, model configuration, training, evaluation, and saving.

        Parameters:
        -----------
        idStr : str, optional
            Column name for site identifier (default 'id').
        sitestratified : str, optional
            Column name for site-level stratification (default 'source').
        stratified : str, optional
            Column name for sample-level stratification (default 'month_date').
        nepoch : int, optional
            Number of training epochs (default 2).
        samplepath : str, optional
            Path for saving/loading sample selections (default 'tmp').
        savePath : str, optional
            Path for saving model outputs (default '/tmp').

        Notes:
        ------
        - Uses bootstrap site selection for spatially independent testing
        - Applies stratified splitting for temporal representation
        - Oversamples extreme values to improve prediction at extremes
        - Configures and trains a physics-constrained neural network
        - Saves models and performance metrics
        """
        # Select test sites using bootstrap approach
        selsites = self.bootstrapSelectSites(idStr=idStr, tpath=samplepath)

        # Split data: training sites vs. independent test sites
        trainsitesIndex = np.where(~self.fulldata[idStr].isin(selsites))[0]
        indTestsitesIndex = np.where(self.fulldata[idStr].isin(selsites))[0]

        # Further split training sites into train/test
        trainIndex, testIndex = self.train_test_splitR(trainsitesIndex, stratified, test_size=0.2)

        # Oversample extreme values in training set
        trainIndex = self.oversampling(trainIndex, 'nox_m', self.th_low, self.th_high, samprop=0.2)

        # Prepare data arrays for model training
        trainX = self.Xn[trainIndex, :]
        trainY = self.yn[trainIndex]
        testX = self.Xn[testIndex, :]
        testY = self.yn[testIndex]
        indtestX = self.Xn[indTestsitesIndex, :]
        indtestY = self.yn[indTestsitesIndex]

        print("Training samples: ", len(trainIndex), '; testing samples: ', len(testIndex),
              ' sited testing samples:', len(indTestsitesIndex))

        # Save dataset indices for reproducibility
        dataindexDict = {'trainIndex': trainIndex, 'testIndex': testIndex, 'indTestsitesIndex': indTestsitesIndex}
        pFile = savePath + '/traintest_index.pkl'
        with open(pFile, 'wb') as fp:  # Pickling
            pickle.dump(dataindexDict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # Set up model output path
        nfeatures = trainX.shape[1]
        mPath = savePath

        # Configure optimizer
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.09, epsilon=1e-3, global_clipnorm=1.5)

        # Configure main model architecture
        nfeas = trainX.shape[1]
        encoders = [1024, 512, 320, 256, 128, 96, 64, 32, 16, 8]
        acts = ['elu' if i == (len(encoders) - 1) else 'relu' for i in range(len(encoders))]
        reg = 'l2'

        # Get normalization parameters for outputs
        scy_mean = self.scy.mean_
        scy_std = self.scy.scale_
        meanpol = scy_mean[0]
        scalepol = scy_std[0]

        # Configure main model hyperparameters
        dact = 'linear'
        bn = True
        dropout = 0.1

        # Configure physics model architecture
        pencoders = [512, 320, 256, 128, 96, 64, 32, 16, 8]
        pacts = ['relu' for i in range(len(pencoders))]
        preg = 'l2'
        pdact = 'linear'
        pisbn = True
        pdropout = 0.1

        # Training attempt counter and limit
        itime = 0
        tlimit = 5
        best_indtest_r2_no2 = -999999999
        best_indtest_r2_nox = -999999999

        # Try training the model (with retry capability)
        while itime < tlimit:
            # Create physics-informed model
            myPhyModel = PhysicsResAutocoderPols(tf_optimizer, nfeas, encoders, acts, meanpol,
                                                 scalepol, dact, bn, dropout, self.logno2max, self.lognoxmax, True,
                                                 pencoders, pacts, pdact, pisbn, pdropout, True, reg, preg)

            # Train the model
            pmetricsHist, best_indtest_r2_no2, best_indtest_r2_nox = myPhyModel.fit2PDE(
                trainX, trainY, testX, testY, indtestX, indtestY, self.scy, None, nepoch, 1588,
                tmppath=mPath + '/train_tmp.csv',
                testdatapath=mPath + '/indtest_bestpre_strict.csv',
                bestmodelpath=mPath
            )

            # Save training history
            histFl = mPath + '/train_hist_strict.csv'
            pmetricsHist.to_csv(histFl, index=False)

            # Get and save both main and physics parameter models
            main_model, para_model = myPhyModel.getModels()
            flmajor = mPath + '/phyend_main'
            self.saveModel(main_model, flmajor)
            flmajor = mPath + '/phyend_para'
            self.saveModel(para_model, flmajor)

            # Exit retry loop
            itime = tlimit

    def ensBaselineModel(self, idStr='id', stratified='month_date', nepoch=2, batch_size=300, samplepath='tmp',
                         savePath='/tmp'):
        """
        Train a baseline Full Residual Neural Network (FRNN) model without physics constraints.

        This method handles the entire training pipeline for the baseline model:
        data splitting, oversampling, model configuration, training, evaluation, and saving.

        Parameters:
        -----------
        idStr : str, optional
            Column name for site identifier (default 'id').
        stratified : str, optional
            Column name for sample-level stratification (default 'month_date').
        nepoch : int, optional
            Number of training epochs (default 2).
        batch_size : int, optional
            Batch size for training (default 300).
        samplepath : str, optional
            Path for saving/loading sample selections (default 'tmp').
        savePath : str, optional
            Path for saving model outputs (default '/tmp').

        Notes:
        ------
        - Uses the same data splitting approach as ensAModel
        - Configures and trains a standard FRNN without physics constraints
        - Saves model performance metrics
        """
        # Select test sites using bootstrap approach
        selsites = self.bootstrapSelectSites(idStr=idStr, tpath=samplepath)

        # Split data: training sites vs. independent test sites
        trainsitesIndex = np.where(~self.fulldata[idStr].isin(selsites))[0]
        indTestsitesIndex = np.where(self.fulldata[idStr].isin(selsites))[0]

        # Further split training sites into train/test
        trainIndex, testIndex = self.train_test_splitR(trainsitesIndex, stratified, test_size=0.2)

        # Oversample extreme values in training set
        trainIndex = self.oversampling(trainIndex, 'nox_m', self.th_low, self.th_high, samprop=0.2)

        # Prepare data arrays for model training
        trainX = self.Xn[trainIndex, :]
        trainY = self.yn[trainIndex]
        testX = self.Xn[testIndex, :]
        testY = self.yn[testIndex]
        indtestX = self.Xn[indTestsitesIndex, :]
        indtestY = self.yn[indTestsitesIndex]

        print("Training samples: ", len(trainIndex), '; testing samples: ', len(testIndex),
              ' sited testing samples:', len(indTestsitesIndex))

        # Configure optimizer
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.09, epsilon=1e-3, global_clipnorm=1.5)

        # Configure model architecture
        nfeas = trainX.shape[1]
        encoders = [1024, 512, 320, 256, 128, 96, 64, 32, 16, 8]
        acts = ['elu' if i == (len(encoders) - 1) else 'relu' for i in range(len(encoders))]

        # Get normalization parameters for outputs
        scy_mean = self.scy.mean_
        scy_std = self.scy.scale_
        meanpol = scy_mean[0]
        scalepol = scy_std[0]

        # Configure hyperparameters
        dact = 'linear'
        isbn = True
        inresidual = True
        dropout = 0.1
        # Create and train baseline FRNN model
        baseFRNN = baselineFRNN(tf_optimizer, nfeas, encoders, acts, meanpol, scalepol, dact, isbn,inresidual)
        pmetricsDf, fhist_res = baseFRNN.fit(
            trainX, trainY, testX, testY, indtestX, indtestY, self.scy, tf_epochs=nepoch,
            batch_size=batch_size, tmppath=savePath, bestmodelpath=savePath
        )

        # Save performance metrics and training history
        outfl = savePath + '/pmetricsDf.csv'
        pmetricsDf.to_csv(outfl, index=False)

    def checkMetrics(self, amodel, inX, iny, flag="", sindex=None, tpath=None):
        """
        Evaluate model performance and optionally save predictions.

        This method calculates R² and RMSE metrics for model predictions
        and can save detailed prediction results to a CSV file.

        Parameters:
        -----------
        amodel : keras.Model
            Model to evaluate.
        inX : numpy.ndarray
            Input features for prediction.
        iny : numpy.ndarray
            True target values.
        flag : str, optional
            Identifier for the dataset being evaluated (default "").
        sindex : numpy.ndarray, optional
            Indices of samples for saving predictions (default None).
        tpath : str, optional
            Path for saving prediction results (default None).

        Returns:
        --------
        tuple
            (r2, rmse) performance metrics

        Notes:
        ------
        - Applies inverse scaling and exponential transform to predictions
        - Calculates R² and RMSE on the original scale
        - If sindex is provided, saves detailed prediction results
        """
        # Generate predictions
        prey = amodel.predict(inX)

        # Apply inverse scaling and transform back to original scale
        prey = np.exp(self.scy.inverse_transform(prey[:, -1]))
        obsy = np.exp(self.scy.inverse_transform(iny))[:, 0]

        # Calculate performance metrics
        r2 = r2np(prey, obsy)
        rmse = rmse2np(prey, obsy)

        # Print results
        print(flag + " ... ... R2:", r2, ";RMSE:", rmse)
        print(prey, obsy)
        print(prey.shape, obsy.shape)

        # Save detailed predictions if requested
        if sindex is not None:
            saveDt = pd.DataFrame({
                'newid2': self.fulldata.iloc[sindex]['newid2'].values,
                'iweek': self.fulldata.iloc[sindex]['iweek'].values,
                'pre': prey,
                'obs': obsy
            }, index=self.fulldata.iloc[sindex].index)

            tfl = tpath + '/' + flag + '_predicted.csv'
            saveDt.to_csv(tfl, index=True, index_label='index')
