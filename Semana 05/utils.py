from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

class ExtendedTensorBoard(TensorBoard):
    """
    Adaptado de https://github.com/tensorflow/tensorflow/issues/31542
    """
    def __init__(self, x, y,log_dir='logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None,
                            **kwargs,):
        self.x=x
        self.y=y
        super(ExtendedTensorBoard, self).__init__(log_dir,
                                                    histogram_freq,
                                                    write_graph,
                                                    write_images,
                                                    update_freq,
                                                    profile_batch,
                                                    embeddings_freq,
                                                    embeddings_metadata,)
    
    def _log_gradients(self, epoch):
        writer = self._get_writer(self._train_run_name)
        with writer.as_default(), tf.GradientTape() as g:
            
            features=tf.constant(self.x)
            y_true=tf.constant(self.y)
            
            y_pred = self.model(features)  # forward-propagation
            loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
#         Sobre-escrevemos essa função da super classe pois necessitamos
#         adicionar a funcionalidade de gravar os gradientes.
#         Como tambem queremos suas funcionalidades originais, tambem invocamos o metodo super       
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)