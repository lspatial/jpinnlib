import tensorflow as tf
import keras.backend as K
from tensorflow.keras.callbacks import Callback

# Custom callback to evaluate additional test data
class EvaluateAdditionalData(Callback):
    def __init__(self, addtestX,addtestY,mean,scale):
        super().__init__()
        self.testX = addtestX
        self.testY = addtestY
        self.mean = mean
        self.scale = scale
        self.r2_metrics = []
        self.rmse_metrics = []

    def r2tensor(self,y_true, y_pred):
        """
        rsquared for regression used in Keras
        :param y_true: array tensor for observation, just the last output .
        :param y_pred: array tensor for predictions, just the last output.
        :return: r2
        """
        y_true_transformed = tf.cast(K.flatten(K.exp(y_true * self.scale + self.mean)),tf.float64)
        y_pred_transformed = tf.cast(K.flatten(K.exp(y_pred * self.scale + self.mean)),tf.float64)
        SS_res =  K.sum(K.square(y_true_transformed - y_pred_transformed))
        SS_tot = K.sum(K.square(y_true_transformed - K.mean(y_true_transformed)))
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    def rmsetensor(self,y_true, y_pred):
        """
        RMSE with scalar mean and scale transformation for Keras.
        :param y_true: Tensor of true values (last output).
        :param y_pred: Tensor of predicted values (last output).
        :param mean: Scalar mean used in the transformation.
        :param scale: Scalar scale factor used in the transformation.
        :return: RMSE
        """
        # Undo the transformations (exponential and scaling)
        y_true_transformed = tf.cast(K.flatten(K.exp(y_true * self.scale + self.mean)),tf.float64)
        y_pred_transformed = tf.cast(K.flatten(K.exp(y_pred * self.scale + self.mean)),tf.float64)

        # Compute Mean Squared Error
        mse = K.mean(K.square(y_true_transformed - y_pred_transformed))
        # Compute RMSE
        rmse = K.sqrt(mse)
        return rmse

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.testX)
        r2_metric = self.r2tensor(self.testY, y_pred)
        rmse_metric = self.rmsetensor(self.testY, y_pred)
        self.r2_metrics.append(r2_metric.numpy())
        self.rmse_metrics.append(rmse_metric.numpy())
        print(f"Independent Testat epoch {epoch + 1}  r2: {r2_metric:.4f}; rmse:{rmse_metric:.4f}")
