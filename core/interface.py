import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from .util import *
from .train import SeriesPredictor
from .dataset import ClassifierDataSet

import pickle



def pca_dataset():
	configs = {"path" : features_path}
	ds = ClassifierDataSet(configs)
	train_data, train_label = ds("train", False)
	pca = decomposition.PCA(n_components=3)
	pca.fit(train_data)
	nparr = pca.transform(train_data)
	df = pd.DataFrame(nparr)
	# df = pd.concat([df, train_label])
	# import pdb; pdb.set_trace()
	df.columns = ['f1', 'f2', 'f3']
	# df['label'] = train_label.iloc[:].values
	return df

class SeriesData():
	"""
	return list of arrays
	it's user's duty to make sure the filepath is ordered
	"""
	def __init__(self, files):
		self.filepaths = files

	def __call__(self):
		def parse_file(filepath):
			df = pd.read_csv(filepath, sep="\s+")
			data = np.array(df.values[:100, 1])
			return data
		input = []
		for file in self.filepaths:
			input.append(parse_file(file))
                return input


class Predictor():
	def __init__(self, model_name):
		self.checkpoint_dir = checkpoint_rnn_dir
		self.model_dir = model_predictor_dir
		self.model_name = model_name



	def __call__(self, data, step):
		self._load()
		self._predict(data, step)
		return self.pred

	def __repr__(self):
		pass

	def _load(self):
		# model = SeriesPredictor()
		# model = model.create_model_instance()
		model = tf.keras.models.load_model(self.model_dir + self.model_name)
		model.load_weights(self.checkpoint_dir+"train-001.hdf5")
		self.model = model

	def _predict(self, data, step):
		pred = []
		history_data = data
		for _ in range(step):
			x = np.array(history_data)

			x = np.reshape(x, (1, x.shape[0], x.shape[1]))
			# import pdb; pdb.set_trace()
			y = self.model.predict(x)
			# shift window
			history_data.pop(0)
			history_data.append(y[0][1])
			pred.extend(y[0][1])

		self.pred = pred


	def _calculate(self):
		pass


if __name__ == '__main__':
	# ranges = range(200,202)
	# files = [util.travel_path_f(i) for i in ranges]
	# s = SeriesData(files)
	# data = s()
	# print(data)
	# import pdb; pdb.set_trace()

	# values = [1,2,3,4]
	# ds = ClassifierDataSet()
	# train_data, _ = ds("train", False)

	# import pdb; pdb.set_trace()
	# c = Classifier(flag = "m", model_name = skl_model_name)
	# print(c(fdata(values)))




	# p = Predictor('rnn.h5')
	# plt.figure()
	# plt.plot(p(data, 10))
	# plt.show()
