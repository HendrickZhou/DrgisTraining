import tensorflow as tf
from tensorflow.keras import layers
from dataset import *
from util import *


class SeriesPredictor():
	def __init__(self):
		self.model = None
		self.val_callbacks = list()
		self.configs = dict()

	def __call__(self):
		self._ds()
		self._build()
		self.model.summary()
		self._callback()
		self._train()
		self._save()

	def create_model_instance(self):
		self._build()
		return self.model

	def _ds(self):
		ds = SequentialDataset()
		ds.cache(val=True)
		self.train_set = ds("train")
		self.valid_set = ds("val")
		self.test_set = ds("test")


	def _build(self):
		model = tf.keras.Sequential([
			layers.LSTM(100, 
				input_shape = (50, 150), 
				return_sequences=True, 
				#recurrent_activation = 'tanh',
				#dropout = 0.2,
				#activation='relu'
				), # if next layer is Dense, dont' use return_sequences!
			layers.LSTM(50,
				#activation='relu', 
				return_sequences=True),
			layers.LSTM(20,
				return_sequences=False),
                        layers.Dense(150),
		])
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
		model.compile(optimizer=optimizer,#'sgd',
              # loss=tf.keras.losses.BinaryCrossentropy(),
              loss = 'mean_squared_error',
              #metrics=['accuracy'],
		)
		self.model = model

	def _callback(self):
		# checkpoints
		cp_callback_train = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_rnn_dir+checkpoint_train_path,
		        moniter = "loss",
		        verbose=1,
		        save_best_only = True,
		        save_weights_only = False,
		        mode = "max",
		        save_freq="epoch")
		cp_callback_val = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_rnn_dir+checkpoint_val_path,
		        moniter = "val_loss",
		        verbose=1,
		        save_best_only = True,
		        save_weights_only = False,
		        mode = "max",
		        save_freq="epoch")

		# tensorboard training data
		tb_callback = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_rnn_path,
		        histogram_freq = 10,
		        write_graph = True,
		        write_images = True,
		        update_freq = 10)

		self.callbacks = [cp_callback_train, cp_callback_val, tb_callback]

	def _train(self):
		self.model.fit(self.train_set.repeat(),
		          validation_data=self.valid_set,
		          validation_freq=1,
		          steps_per_epoch = 40,
		          epochs=200,
                  	callbacks = self.callbacks,
		          )
	def _save(self):
		self.model.save(model_predictor_dir+"rnn.h5")



if __name__ == '__main__':
	PM = SeriesPredictor()
	PM()
