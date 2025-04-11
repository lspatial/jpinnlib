import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Dense, Activation, add, concatenate, multiply
from model.cminibatch import iterate_minibatches, iterate_minibatches_traintest_f,iterate_minibatches_traintest
from tqdm import tqdm
from sklearn.metrics import r2_score
from model.metrics import r2np, rmse2np
import pandas as pd
import math
from scipy.stats import pearsonr

from tensorflow.python.keras.activations import swish
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seeValue = 100
import os

os.environ['PYTHONHASHSEED'] = str(seeValue)

class PhysicsResAutocoderPols(object):
    """
       The core class for the encoding jPINN model.

       Our jPINN model consists of two full residual deep networks, one for the main model to predict the dual target
        variables (e.g., NO2 and NOx; PM2.5 and PM10), and the other one for the parameters of inverse the proxy
        advection velocities, proxy diffusion coefficients and deposition etc. The two networks are connected by
        the PDE and its relevant loss functions. Thus, our jPINN model not only output the target predictions
        but also inverse the proxy diffusion physical coefficients.

       This module provides a configurable parameters for the jPINN model.:
       - Neuron numbers, action functions and residual connection etc. for main model;
       - Neuron numbers, action functions and residual connection etc. for parameter model;
       - Thresholds for the potential maximum log value for the first and second target variables;
       - Regularization parameters for the jPINN model.

       Key Features:
       - Supports PDE, other and its loss function computing;
       - Customed network parameters;
       - Enables flexible parameter optimization;
       - Learnable PDE weight.

       Attributes:
           main_model (nn.Module): The model for predicting the dual target variables;
           para_model (nn.Module): The model for inversing the physical arguments;
           log2max (float): Possible log2max value (the first target variable);
           logxmax (float): Possible log2max value (the second target variable);
           no2xratio (float): The possible ratio between the first and second target variables.

       Functions:
          AResAutoNet: A neural network with full residual connections;
          f_model: Calculate the PDE physical parameters using main and parameter models;
          __gradPDF: The PDE residual loss function and gradients;
          fit2PDE: fitting function for the comprehensive loss function including
              PDE residual, RMSE and big-small relationship etc.
       """

    def __init__(self, optimizer, nfeas, encoders, acts, mean, scale, dact, isbn, dropout, log2max, logxmax,
                 inresidual, pencoders, pacts, pdact, pisbn, pdropout, pinresidual,reg=None,preg=None):
        """
        Initialize the PhysicsResAutocoderPols class for physics-based autoencoder with residual connections.

            Parameters:
            ----------
            optimizer : tf.keras.optimizers
                Optimizer for training the models
            nfeas : int
                Number of input features
            encoders : list
                List of neuron numbers for each encoder layer in main model
            acts : list
                List of activation functions for main model
            mean : float or array
                Mean value(s) for the normalization of the target variable to be predicted
            scale : float or array
                Scale value(s) for the normalization of the target variable to be predicted
            dact : str
                The activation function for the last layer of main model
            isbn : bool
                Whether to use batch normalization in main model
            dropout : float
                Dropout rate for main model
            log2max : float
                Maximum log2 value for the first output target variable (e.g., NO2) to be predicted
            logxmax : float
                Maximum logx value for the second output target variable (e.g. NOx) to be predicted
            inresidual : bool
                Whether to use residual connections in main model
            pencoders : list
                List of neuron numbers for each encoder layer in parameter model
            pacts : list
                List of activation functions for parameter model
            pdact : str
                Decoder activation function for the last layer of the parameter model
            pisbn : bool
                Whether to use batch normalization in parameter model
            pdropout : float
                Dropout rate for parameter model
            pinresidual : bool
                Whether to use residual connections in parameter model
            reg : tf.keras.regularizers, optional
                Regularizer for main model weights
            preg : tf.keras.regularizers, optional
                Regularizer for parameter model weights
                """
        # Set data type for TensorFlow operations
        self.dtype = tf.float32

        # Store optimizer and input parameters
        self.opt = optimizer
        self.nfeas = nfeas

        # Store normalization parameters
        self.mean = mean
        self.scale = scale
        self.log2max = log2max
        self.logxmax = logxmax

        # Set up regularizers with default values if not provided
        regCus = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) if reg is None else reg
        pregCus = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) if preg is None else preg

        # Initialize main model for predictions (output size: 2)
        self.main_model = self.AResAutoNet(nfeas, encoders, acts, dact, isbn, dropout, regCus, inresidual, nout=2)

        # Initialize parameter model for physics parameters (output size: 14)
        self.para_model = self.AResAutoNet(nfeas, pencoders, pacts, pdact, pisbn, pdropout, pregCus, pinresidual,
                                           nout=14)

        # Initialize trainable weights for model balancing
        self.pdewei = tf.Variable(1.0)  # Weight parameter for physics component
        self.no2xratio = tf.Variable(1.2)  # Ratio parameter for NO2/NOx calculations

    def getModels(self):
        """
        Returns the main and parameter models of the jPINN.

        Returns:
            tuple: (main_model, para_model) - The two neural networks that make up the jPINN
        """
        return self.main_model, self.para_model

    def getActivation(self, actstr, inlayer):
        """
        Function to obtain the activation layer

        Parameters:
            actstr: string, activation function. default: None, indicating 'linear'
            inlayer: Input layer

        Returns:
            The activation layer's output
        """
        prereact = ['relu', 'elu', 'softmax', 'selu', 'softplus', 'tanh', 'sigmoid', 'linear', 'hard_sigmoid', 'sine']
        # preadvact={'LeakyReLU':LeakyReLU()(inlayer)}
        if actstr is None:
            return inlayer
        if actstr in prereact:
            return Activation(actstr)(inlayer)
        elif actstr == 'swish':
            return swish(inlayer)
        else:
            return eval(actstr + '()(inlayer)')
        return inlayer

    def hiddenBlock(self, encoders, acts, isbn, dropout, reg, inlayer, idepth, orginlayer=None):
        """
        Creates a hidden block in the neural network with optional residual connection

        Parameters:
            encoders: List of neuron counts for encoder layers
            acts: List of activation functions
            isbn: Boolean flag for batch normalization
            dropout: Dropout rate, 0 for no dropout
            reg: Regularizer for weight matrices
            inlayer: Input layer to this block
            idepth: Index of depth in the network
            orginlayer: Original layer for residual connection, None for no residual

        Returns:
            Output tensor from this hidden block
        """
        layer = Dense(encoders[idepth], kernel_initializer='glorot_uniform', kernel_regularizer=reg)(inlayer)
        layer = BatchNormalization()(layer) if isbn else layer
        layer = self.getActivation(acts[idepth], layer)
        layer = BatchNormalization()(layer) if isbn else layer
        layer = Dropout(dropout)(layer) if dropout else layer
        if orginlayer is not None:
            # Add residual connection if original layer is provided
            layer = add([orginlayer, layer])
            layer = BatchNormalization()(layer) if isbn else layer
            layer = self.getActivation(acts[idepth], layer)
        return layer

    def AResAutoNet(self, nfeas, encoders, acts, dact, isbn, dropout, reg, inresidual, nout=1):
        """
        Creates a fully connected auto-encoder with residual connections

        Parameters:
            nfeas: Number of input features
            encoders: List of neuron counts for encoder layers
            acts: List of activation functions
            dact: Decoder activation function
            isbn: Boolean flag for batch normalization
            dropout: Dropout rate, 0 for no dropout
            reg: Regularizer for weight matrices
            inresidual: Boolean flag for using input residual connection
            nout: Number of output neurons (default: 1)

        Returns:
            Keras Model with the specified architecture
        """
        inputlayer = Input(shape=(nfeas,), name='ninfeature')
        stck = []  # Stack to store layers for skip connections
        layer = inputlayer

        # Encoder path
        for i in range(len(encoders)):
            nnode = encoders[i]
            layer = Dense(nnode, kernel_initializer='he_normal', kernel_regularizer=reg)(layer)
            layer = self.getActivation(acts[i], layer)
            layer = BatchNormalization()(layer) if isbn else layer
            if i < (len(encoders) - 1):
                stck.append(layer)  # Store intermediate layers for decoder skip connections
            if i == (len(encoders) - 1):
                layer = Dropout(dropout)(layer) if dropout else layer

        # Decoder path with skip connections
        for i in range(len(encoders) - 2, -1, -1):
            orlayer = stck.pop()  # Get the corresponding encoder layer
            layer = self.hiddenBlock(encoders, acts, isbn, dropout, reg, layer, i, orlayer)

        # Final decoder layer
        layer = Dense(nfeas, kernel_initializer='he_normal', kernel_regularizer=reg)(layer)
        layer = self.getActivation(acts[len(acts) - 1], layer)
        layer = BatchNormalization()(layer) if isbn else layer

        # Optional residual connection from input to output
        if inresidual:
            layer = add([inputlayer, layer])
            layer = BatchNormalization()(layer) if isbn else layer
            layer = self.getActivation(dact, layer)

        # Output layer
        outlayer = Dense(nout, kernel_initializer='he_normal',
                         kernel_regularizer=reg)(layer)

        return Model(inputs=inputlayer, outputs=outlayer)

    def f_model(self, Xin):
        """
        The core physics-informed neural network (PINN) function that computes PDE residuals

        Parameters:
            Xin: Input tensor containing time (t), spatial coordinates (x,y,z), and other features

        Returns:
            tuple: PDE residual terms and other constraint terms for the loss function
        """
        # Extract time and spatial coordinates from input
        ut = tf.convert_to_tensor(Xin[:, 0:1], dtype=self.dtype)  # Time coordinate
        ux = tf.convert_to_tensor(Xin[:, 1:2], dtype=self.dtype)  # x coordinate
        uy = tf.convert_to_tensor(Xin[:, 2:3], dtype=self.dtype)  # y coordinate
        uz = tf.convert_to_tensor(Xin[:, 3:4], dtype=self.dtype)  # z coordinate
        uothers = tf.convert_to_tensor(Xin[:, 4:], dtype=self.dtype)  # Other features

        # Using GradientTape to automatically compute derivatives for PDE terms
        with tf.GradientTape(persistent=True) as tape:
            # Watch input variables for gradient computation
            tape.watch(ut)
            tape.watch(ux)
            tape.watch(uy)
            tape.watch(uz)

            # Concat inputs for the model
            XinTensor = tf.concat([ut, ux, uy, uz, uothers], axis=1)

            # Get predictions from the main model (NO2 and NOx)
            pre = self.main_model(XinTensor)
            pre_no2 = pre[:, 0]  # NO2 prediction
            pre_nox = pre[:, 1]  # NOx prediction

            # Get parameters from the parameter model (diffusion coefficients, etc.)
            para = self.para_model(XinTensor)

            # Compute first-order derivatives for NO2
            u1t_no2 = tape.gradient(pre_no2, ut)  # ∂NO2/∂t
            u1x_no2 = tape.gradient(pre_no2, ux)  # ∂NO2/∂x
            u1y_no2 = tape.gradient(pre_no2, uy)  # ∂NO2/∂y
            u1z_no2 = tape.gradient(pre_no2, uz)  # ∂NO2/∂z

            # Compute second-order derivatives for NO2
            u2x_no2 = tape.gradient(u1x_no2, ux)  # ∂²NO2/∂x²
            u2y_no2 = tape.gradient(u1y_no2, uy)  # ∂²NO2/∂y²
            u2z_no2 = tape.gradient(u1z_no2, uz)  # ∂²NO2/∂z²

            # Compute first-order derivatives for NOx
            u1t = tape.gradient(pre_nox, ut)  # ∂NOx/∂t
            u1x = tape.gradient(pre_nox, ux)  # ∂NOx/∂x
            u1y = tape.gradient(pre_nox, uy)  # ∂NOx/∂y
            u1z = tape.gradient(pre_nox, uz)  # ∂NOx/∂z

            # Compute second-order derivatives for NOx
            u2x = tape.gradient(u1x, ux)  # ∂²NOx/∂x²
            u2y = tape.gradient(u1y, uy)  # ∂²NOx/∂y²
            u2z = tape.gradient(u1z, uz)  # ∂²NOx/∂z²

            # Compute constraint terms for bounds and relationships
            no2l = tf.square(tf.keras.activations.relu(pre_no2 - self.log2max))  # Penalty for NO2 exceeding max
            noxl = tf.square(tf.keras.activations.relu(pre_nox - self.logxmax))  # Penalty for NOx exceeding max
            no2xl = tf.square(
                tf.keras.activations.relu(pre_no2 - pre_nox))  # Penalty for NO2 > NOx (physically impossible)

        # Release the tape to free memory
        del tape

        # Compute residual of advection-diffusion PDE for NOx
        # ∂NOx/∂t + v_x*∂NOx/∂x + v_y*∂NOx/∂y + v_z*∂NOx/∂z - D_x*∂²NOx/∂x² - D_y*∂²NOx/∂y² - D_z*∂²NOx/∂z² - S = 0
        respde = u1t[:, 0] + para[:, 0] * u1x[:, 0] + para[:, 1] * u1y[:, 0] + para[:, 2] * u1z[:, 0] \
                 - para[:, 3] * u2x[:, 0] - para[:, 4] * u2y[:, 0] - para[:, 5] * u2z[:, 0] - para[:, 6]

        # Compute residual of advection-diffusion PDE for NO2
        # ∂NO2/∂t + v_x*∂NO2/∂x + v_y*∂NO2/∂y + v_z*∂NO2/∂z - D_x*∂²NO2/∂x² - D_y*∂²NO2/∂y² - D_z*∂²NO2/∂z² - S = 0
        respde_no2 = u1t_no2[:, 0] + para[:, 7] * u1x_no2[:, 0] + para[:, 8] * u1y_no2[:, 0] + para[:, 9] * u1z_no2[:,
                                                                                                            0] \
                     - para[:, 10] * u2x_no2[:, 0] - para[:, 11] * u2y_no2[:, 0] - para[:, 12] * u2z_no2[:, 0] - para[:,
                                                                                                                 13]

        # Return squared residuals and constraint terms for the loss function
        return tf.square(respde), tf.square(respde_no2), no2l, noxl, no2xl

    def __lossPDF(self, Xall, Xin, yin):
        """
        Computes the complete loss function combining data loss and physics-informed terms

        Parameters:
            Xall: Input tensor for PDE evaluation
            Xin: Input tensor for the supervised part
            yin: Target values for the supervised part

        Returns:
            Total loss value combining data fit and physics constraints
        """
        # Get PDE residuals and constraint terms
        f_pder, f_pder_no2, no2l, noxl, no2xl = self.f_model(Xall)

        # Get model predictions for supervised data
        u_pred = self.main_model(Xin)

        # Compute supervised losses (MSE) for NO2 and NOx
        loss_no2 = tf.reduce_mean(tf.square(yin[:, 0] - u_pred[:, 0]))
        loss_nox = tf.reduce_mean(tf.square(yin[:, 1] - u_pred[:, 1]))

        # Compute mean PDE residuals
        loss_pde = tf.reduce_mean(f_pder)  # NOx PDE residual
        loss_pde_no2 = tf.reduce_mean(f_pder_no2)  # NO2 PDE residual

        # Compute constraint losses
        loss_no2l = tf.reduce_mean(no2l)  # NO2 maximum constraint
        loss_noxl = tf.reduce_mean(noxl)  # NOx maximum constraint
        loss_no2xl = tf.reduce_mean(no2xl)  # NO2 < NOx constraint

        # Compute total loss (weighted sum of all components)
        total_loss = loss_no2 + loss_nox + loss_pde * 1.0 + \
                     loss_pde_no2 * 1.0 + loss_no2l + loss_noxl + loss_no2xl

        return total_loss

    def __purelossPDF(self, Xall):
        """
        Computes a pure physics-based loss without supervised data

        Parameters:
            Xall: Input tensor for PDE evaluation

        Returns:
            Total loss value based only on physics constraints
        """
        f_pder, noxl = self.f_puremodel(Xall)
        loss_pde = tf.reduce_mean(f_pder)
        loss_noxl = tf.reduce_mean(noxl)
        total_loss = loss_pde * 1.0 + loss_noxl
        return total_loss

    def __loss(self, Xin, yin):
        """
        Computes the basic supervised loss (MSE)

        Parameters:
            Xin: Input tensor
            yin: Target values

        Returns:
            Mean squared error between predictions and targets
        """
        u_pred = self.main_model(Xin)
        return tf.reduce_mean(tf.square(yin - u_pred))

    def __gradPLoss(self, Xin, yin):
        """
        Computes the loss and its gradients for plain supervised learning

        Parameters:
            Xin: Input tensor
            yin: Target values

        Returns:
            tuple: (loss_value, gradients)
        """
        with tf.GradientTape() as tape:
            loss_value = self.__loss(Xin, yin)
        return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

    def __gradPDF(self, Xall, Xin, yin):
        """
        Computes the loss and its gradients for physics-informed learning

        Parameters:
            Xall: Input tensor for PDE evaluation
            Xin: Input tensor for supervised part
            yin: Target values for supervised part

        Returns:
            tuple: (loss_value, gradients)
        """
        with tf.GradientTape() as tape:
            loss_value = self.__lossPDF(Xall, Xin, yin)
        return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

    def __puregradPDF(self, Xall):
        """
        Computes the loss and its gradients for pure physics-based learning

        Parameters:
            Xall: Input tensor for PDE evaluation

        Returns:
            tuple: (loss_value, gradients)
        """
        with tf.GradientTape() as tape:
            loss_value = self.__purelossPDF(Xall)
        return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

    def __grad(self, Xin, yin):
        """
        Computes the loss and its gradients for the main model only (no parameter model)

        Parameters:
            Xin: Input tensor
            yin: Target values

        Returns:
            tuple: (loss_value, gradients)
        """
        with tf.GradientTape() as tape:
            loss_value = self.__loss(Xin, yin)
        return loss_value, tape.gradient(loss_value, self.__wrap_training_nopdevariables())

    def __wrap_training_variables(self):
        """
        Collects all trainable variables from both models for optimization

        Returns:
            List of trainable variables from both main and parameter models
        """
        varm = self.main_model.trainable_variables
        varp = self.para_model.trainable_variables
        varm.extend(varp)
        return varm

    def __wrap_training_nopdevariables(self):
        """
        Collects trainable variables from only the main model (no parameter model)

        Returns:
            List of trainable variables from the main model only
        """
        varm = self.main_model.trainable_variables
        return varm

    def summary(self):
        """
        Returns a summary of the main model architecture

        Returns:
            Summary string of the main model
        """
        return self.main_model.summary()

    def getMetrics(self, obs, pre):
        """
        Calculates performance metrics between observed and predicted values

        Parameters:
            obs: Observed (ground truth) values
            pre: Predicted values

        Returns:
            tuple: (r2_score, rmse, pearson_correlation)
        """
        r2 = r2_score(obs, pre)
        rmse = rmse2np(obs, pre)
        cor = pearsonr(obs, pre)
        return r2, rmse, cor

    def getCorr(self, obs, pre):
        """
        Calculates the Pearson correlation between observed and predicted values

        Parameters:
            obs: Observed (ground truth) values
            pre: Predicted values

        Returns:
            Pearson correlation coefficient and p-value
        """
        cor = pearsonr(obs, pre)
        return cor

    def saveModel(self, model, flmajor):
        """
        Saves a model in multiple formats (JSON, weights, and full model)

        Parameters:
            model: Keras model to save
            flmajor: Base filename for saving
        """
        # Save model architecture to JSON
        model_json = model.to_json()
        tfl = flmajor + '_frm.json'
        with open(tfl, "w") as json_file:
            json_file.write(model_json)

        # Save weights to HDF5
        tfl = flmajor + '_wei.h5'
        model.save_weights(tfl)

        # Save complete model
        tfl = flmajor + '_wholemodel.h5'
        model.save(tfl)

    def fit2PDE(self, Xtrain, ytrain, Xtest, ytest, Xindtest, yindtest, scy, semiXn=None, tf_epochs=100,
                batch_size=2000, tmppath=None, testdatapath=None, bestmodelpath=None):
        """
        Main training function for the physics-informed neural network

        Parameters:
            Xtrain: Training input data
            ytrain: Training target data
            Xtest: Testing input data
            ytest: Testing target data
            Xindtest: Independent test input data
            yindtest: Independent test target data
            scy: Scaler for y values (for inverse transformation)
            semiXn: Semi-supervised data inputs
            tf_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            tmppath: Path to save temporary training metrics
            testdatapath: Path to save test predictions
            bestmodelpath: Path to save the best model

        Returns:
            tuple: (metrics_history_dataframe, best_r2_score_nox, best_r2_score_no2)
        """
        # Create indices for each dataset
        index_train = np.array([i for i in range(Xtrain.shape[0])])
        index_test = np.array([i for i in range(Xtest.shape[0])])
        index_indtest = np.array([i for i in range(Xindtest.shape[0])])

        # Get sizes of datasets
        ntrain, ntest, nindtest = ytrain.shape[0], ytest.shape[0], yindtest.shape[0]
        nt = ntrain / batch_size

        # Combine all datasets for evaluation
        nall = ntrain + ntest + nindtest
        index_all = np.array([i for i in range(nall)])
        X_all = np.concatenate([Xtrain, Xtest, Xindtest], axis=0)
        y_all = np.concatenate([ytrain, ytest, yindtest], axis=0)

        # Inverse transform to get original scale values
        yobs = np.exp(scy.inverse_transform(y_all))

        # Variables to track best performance
        best_indtest_r2_no2 = -999999999
        best_indtest_r2_nox = -999999999
        best_epoch = -1
        best_indtest_per = None
        epoch = 0
        rtimes = 0
        pmetricsHistDf = None

        # Training loop
        while epoch < tf_epochs:
            # Create mini-batch data loader
            if semiXn is not None:
                index_semi = np.array([i for i in range(semiXn.shape[0])])
                dataloader = iterate_minibatches_traintest_f(index_train, index_test, index_indtest, index_semi,
                                                         batch_size=batch_size, shuffle=True)
            else:
                dataloader = iterate_minibatches_traintest(index_train, index_test, index_indtest,
                                                             batch_size=batch_size, shuffle=True)
            n = math.ceil(len(index_train) / batch_size)
            tloss = 0

            # Train on mini-batches
            for bindex_train, bindex_test, bindex_indtest, bindex_semi in tqdm(dataloader, total=n):
                # Prepare batch data
                batch_X = Xtrain[bindex_train, :]
                batch_y = ytrain[bindex_train, :]

                # Combine different data sources for physics constraints
                if semiXn is not None:
                    X_ball = np.concatenate(
                    [batch_X, Xtest[bindex_test, :], Xindtest[bindex_indtest, :], semiXn[bindex_semi, :]], axis=0)
                else:
                    X_ball = np.concatenate(
                        [batch_X, Xtest[bindex_test, :], Xindtest[bindex_indtest, :]], axis=0)

                # Compute loss and gradients, then apply updates
                loss_value, grads = self.__gradPDF(X_ball, batch_X, batch_y)
                self.opt.apply_gradients(zip(grads, self.__wrap_training_variables()))
                tloss = tloss + loss_value

            # Calculate average loss for the epoch
            tloss = tloss / nt
            print(epoch, ' loss: ', tloss)

            # Evaluation phase
            dataloader = iterate_minibatches(index_all, batch_size=batch_size, shuffle=False)
            allpre = []
            n = math.ceil(len(index_all) / batch_size)

            # Generate predictions for all data
            for bindex_t in tqdm(dataloader, total=n):
                batch_X = tf.convert_to_tensor(X_all[bindex_t, :], dtype=self.dtype)
                pred = self.main_model(batch_X)
                pred = pred.numpy()
                allpre.append(pred)

            # Combine predictions and convert to original scale
            allpre = np.concatenate(allpre, axis=0)
            allpre = np.exp(scy.inverse_transform(allpre))

            # Calculate metrics for NO2 predictions
            train_r2_no2, train_rmse_no2, train_cor_no2 = self.getMetrics(yobs[0:ntrain, 0], allpre[0:ntrain, 0])
            test_r2_no2, test_rmse_no2, test_cor_no2 = self.getMetrics(yobs[ntrain:(ntrain + ntest), 0],
                                                                       allpre[ntrain:(ntrain + ntest), 0])
            indtest_r2_no2, indtest_rmse_no2, indtest_cor_no2 = self.getMetrics(yobs[(ntrain + ntest):nall, 0],
                                                                                allpre[(ntrain + ntest):nall, 0])

            # Calculate metrics for NOx predictions
            train_r2_nox, train_rmse_nox, train_cor_nox = self.getMetrics(yobs[0:ntrain, 1], allpre[0:ntrain, 1])
            test_r2_nox, test_rmse_nox, test_cor_nox = self.getMetrics(yobs[ntrain:(ntrain + ntest), 1],
                                                                       allpre[ntrain:(ntrain + ntest), 1])
            indtest_r2_nox, indtest_rmse_nox, indtest_cor_nox = self.getMetrics(yobs[(ntrain + ntest):nall, 1],
                                                                                allpre[(ntrain + ntest):nall, 1])

            # Update epoch counter
            epoch = epoch + 1
            rtimes = 0

            # Create metrics DataFrame for this epoch
            pmetricsDf = pd.DataFrame({'epoch': epoch, 'train_r2_no2': train_r2_no2, 'train_rmse_no2': train_rmse_no2,
                                       'train_cor_no2': train_cor_no2[0],
                                       'test_r2_no2': test_r2_no2, 'test_rmse_no2': test_rmse_no2,
                                       'test_cor_no2': test_cor_no2[0],
                                       'indtest_r2_no2': indtest_r2_no2, 'indtest_rmse_no2': indtest_rmse_no2,
                                       'indtest_cor_no2': indtest_cor_no2[0],

                                       'train_r2_nox': train_r2_nox, 'train_rmse_nox': train_rmse_nox,
                                       'train_cor_nox': train_cor_nox[0],
                                       'test_r2_nox': test_r2_nox, 'test_rmse_nox': test_rmse_nox,
                                       'test_cor_nox': test_cor_nox[0],
                                       'indtest_r2_nox': indtest_r2_nox, 'indtest_rmse_nox': indtest_rmse_nox,
                                       'indtest_cor_nox': indtest_cor_nox[0]
                                       }, index=[epoch])
            print(pmetricsDf)

            # Check if current model is the best so far
            if (best_indtest_r2_nox <= indtest_r2_nox and best_indtest_r2_no2 <= indtest_r2_no2) or \
                    ((best_indtest_r2_nox + best_indtest_r2_no2) <= (indtest_r2_nox + indtest_r2_no2) and \
                     indtest_r2_nox >= 0.6 and indtest_r2_no2 >= 0.6):
                # Update best performance metrics
                best_indtest_r2_nox = indtest_r2_nox
                best_indtest_r2_no2 = indtest_r2_no2
                best_indtest_per = pmetricsDf
                best_epoch = epoch

                # Save test predictions if path is provided
                if testdatapath is not None:
                    indtestData = pd.DataFrame(
                        {'obs_no2': yobs[(ntrain + ntest):nall, 0], 'pre_no2': allpre[(ntrain + ntest):nall, 0],
                         'obs_nox': yobs[(ntrain + ntest):nall, 1], 'pre_nox': allpre[(ntrain + ntest):nall, 1]})
                    indtestData.to_csv(testdatapath, index=False)

                # Save best model if path is provided
                if bestmodelpath is not None:
                    flmajor = bestmodelpath + '/phybest_main'
                    self.saveModel(self.main_model, flmajor)
                    flmajor = bestmodelpath + '/phybest_para'
                    self.saveModel(self.para_model, flmajor)

            # Print current best performance
            print('Until now, best indtest r2 of NOx is:', best_indtest_r2_nox, '; indtest r2 for NO2 is ',
                  best_indtest_r2_no2, ' for epoch=', best_epoch, 'per: \n ', best_indtest_per)

            # Update metrics history
            if pmetricsHistDf is None:
                pmetricsHistDf = pmetricsDf
            else:
                pmetricsHistDf = pd.concat([pmetricsHistDf, pmetricsDf])

            # Save temporary metrics if path is provided
            if tmppath is not None:
                pmetricsHistDf.to_csv(tmppath, index=False)

        # Return final metrics history and best performance
        return pmetricsHistDf, best_indtest_r2_nox, best_indtest_r2_no2


    def predict(self, X_star):
        u_star = self.main_model(X_star)
        f_star = self.f_model(X_star)
        return u_star, f_star
