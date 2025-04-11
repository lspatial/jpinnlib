import numpy as np
from model.metrics import r2_factory,rmse_factory
from model.addevaluation import EvaluateAdditionalData
from keras.callbacks import ModelCheckpoint

from model.metrics import r2np, rmse2np

import os
import pandas as pd
import keras
import resautonet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seeValue = 100
import os

os.environ['PYTHONHASHSEED'] = str(seeValue)


class baselineFRNN(object):
    """
    Full Residual Neural Network baseline implementation.
    This class implements a residual autoencoder for regression tasks with customizable
    architecture and regularization options.
    """

    def __init__(self, optimizer, nfeas, encoders, acts, mean, scale, dact, isbn, inresidual, reg=None):
        """
        Initialize the FRNN model.

        Parameters:
        -----------
        optimizer : str or optimizer object
            The optimizer to use for training (e.g. "adam", "sgd").
        nfeas : int
            Number of input features.
        encoders : list
            List of integers representing the number of neurons in each encoder layer.
        acts : list
            List of activation functions for each layer.
        mean : float or array
            Mean values used for data normalization.
        scale : float or array
            Scale values used for data normalization.
        dact : str
            Default activation function if not specified in acts.
        isbn : bool
            Whether to use batch normalization.
        inresidual : bool or list
            Specifies whether to use residual connections and where.
        reg : keras.regularizers, optional
            Regularization to apply (default is None, which uses L1L2 with 0 coefficients).
        """
        self.optimizer = optimizer
        self.nfeas = nfeas

        # Set default regularization if none provided
        if reg is None:
            reg = keras.regularizers.l1_l2(0)

        # Create the residual autoencoder model
        modelCls = resautonet.model.resAutoencoder(
            nfea=nfeas,  # Number of input features
            layernNodes=encoders,  # Hidden layer architecture
            acts=acts,  # Activation functions
            extranOutput=1,  # Number of tasks (outputs)
            inresidual=inresidual,  # Residual connection configuration
            reg=reg,  # Regularization
            batchnorm=isbn,  # Batch normalization flag
            outnres=None,  # Output residual connections
            defact=dact,  # Default activation function
            outputtype=0  # Output layer type
        )

        # Build the model
        self.model = modelCls.resAutoNet()
        self.model.summary()

        # Store normalization parameters
        self.mean = mean
        self.scale = scale

        # Compile the model with metrics
        # Note: Custom metrics r2KAuto and r2K are used alongside MSE
        r2_metric = r2_factory(mean=mean, scale=scale)
        rmse_metric = rmse_factory(mean=mean, scale=scale)
        self.model.compile(
            optimizer="adam",  # Using adam regardless of input optimizer?
            loss='mean_squared_error',
            metrics=[r2_metric, rmse_metric]
        )

    def getmetrics(self, ypre, ylab, scy):
        """
        Calculate performance metrics (R² and RMSE) after inverse transforming predictions.

        Parameters:
        -----------
        ypre : numpy.ndarray
            Predicted values (in normalized space).
        ylab : numpy.ndarray
            True labels (in normalized space).
        scy : scaler object
            Scaler used for output normalization that has inverse_transform method.

        Returns:
        --------
        r2_train : float
            R-squared coefficient.
        rmse_train : float
            Root mean squared error.
        """
        # Convert normalized outputs back to original scale and apply exponential transform
        obs = np.exp(scy.inverse_transform(ylab))
        prenores = np.exp(scy.inverse_transform(ypre))  # Take last column for prediction

        # Calculate metrics
        r2_train = r2np(obs, prenores)
        rmse_train = rmse2np(obs, prenores)

        return r2_train, rmse_train

    def fit(self, Xtrain, ytrain, Xtest, ytest, Xindtest, yindtest, scy, tf_epochs=2,
                 batch_size=100, tmppath=None, bestmodelpath=None):
        """
        Train the FRNN model and evaluate performance.

        Parameters:
        -----------
        Xtrain : numpy.ndarray
            Training features.
        ytrain : numpy.ndarray
            Training labels.
        Xtest : numpy.ndarray
            Testing features.
        ytest : numpy.ndarray
            Testing labels.
        Xindtest : numpy.ndarray
            Independent test set features.
        yindtest : numpy.ndarray
            Independent test set labels.
        scy : scaler object
            Output scaler for inverse transformations.
        tf_epochs : int, optional
            Number of training epochs (default 2).
        batch_size : int, optional
            Batch size for training (default 100).
        tmppath : str, optional
            Path to save metrics DataFrame (default None).
        bestmodelpath : str, optional
            Path to save best model weights (default None).

        Returns:
        --------
        pmetricsDf : pandas.DataFrame
            DataFrame containing performance metrics.
        fhist_res : keras.History
            Training history object.
        """
        # Setup checkpoint to save best model based on training loss
        checkpointw = ModelCheckpoint(
            bestmodelpath,
            monitor="loss",
            verbose=0,
            save_best_only=True,
            mode="min"
        )

        indtestmetrics_callback=EvaluateAdditionalData( Xindtest, yindtest,self.mean,self.scale)
        # Train the model
        fhist_res = self.model.fit(
            Xtrain, ytrain,
            batch_size=batch_size,
            epochs=tf_epochs,
            verbose=2,
            shuffle=True,
            validation_data=(Xtest, ytest),
            callbacks=[checkpointw,indtestmetrics_callback]
        )
        trainHist=pd.DataFrame(fhist_res.history)
        trainHist['epoch']=[i for i in range(1,tf_epochs+1)]
        trainHist['ind_test_r2']=indtestmetrics_callback.r2_metrics
        trainHist['ind_test_rmse']=indtestmetrics_callback.rmse_metrics
        # Evaluate on training set
        y_train_pred = self.model.predict(Xtrain)  # This should be self.model.predict() instead
        r2_train, rmse_train = self.getmetrics(y_train_pred, ytrain, scy)

        # Evaluate on test set
        y_test_pred = self.model.predict(Xtest)  # This should be self.model.predict() instead
        r2_test, rmse_test = self.getmetrics(y_test_pred, ytest, scy)

        # Evaluate on independent test set
        y_indtest_pred = self.model.predict(Xindtest)  # This should be self.model.predict() instead
        r2_indtest, rmse_indtest = self.getmetrics(y_indtest_pred, yindtest, scy)

        # Compile metrics into DataFrame
        pmetricsDf = pd.DataFrame({
            'train_r2': r2_train,
            'train_rmse': rmse_train,
            'test_r2': r2_test,
            'test_rmse': rmse_test,
            'indtest_r2': r2_indtest,
            'indtest_rmse': rmse_indtest
        }, index=[0])

        # Save metrics if path provided
        if tmppath is not None:
            tmppathFl=tmppath+'/pmetricsDf.csv'
            pmetricsDf.to_csv(tmppathFl, index=False)
            tmppathFl = tmppath + '/train_tmp.csv'
            trainHist.to_csv(tmppathFl, index=False)

        return pmetricsDf, trainHist
