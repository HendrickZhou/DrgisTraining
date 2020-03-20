"""
step1: task data preperation
step2: save dataset meta info as dataset, and train
step3: use tfrecord as dataset
"""
import tensorflow as tf 
from tensorflow import feature_column
import pandas as pd
from sklearn.model_selection import train_test_split
from util import *

class SequentialDataset():
	"""
	only sequential data need files as input
	"""
	def init_resource_loc(self):
		default_path_f = travel_path_f
		self.default_path = [travel_path_f(i) for i in range(1, 2001)]	

	def __init__(self, configs=None):
		self.init_resource_loc()
		if configs is None:
			configs = dict()
			configs["path"] = self.default_path

		if "path" not in configs:
			configs["path"] = self.default_path

		self.configs = configs


	def cache(self, fold = 10, k = 8, val = False):
		self._parse_args()
		self.ds = self._split(fold, k, val)
		self.cached = True

	def __call__(self, flag="train"):
		if not self.cached:
			print("cache it first")
			return
		if flag is "train":
			return self._pipeline(self.ds[0])
		elif flag is "test":
			return self._pipeline(self.ds[1])
		elif flag is "val":
			return self._pipeline(self.ds[2]) # use at your own risk
		else:
			raise Exception("invalid parameter flag, should be train, test or val")


	def __repr__(self):
		doc = "We assume the dataset is named properly after this pattern:\n"
		doc = doc + "***-(decimal number).suffix"
		doc = doc + "the filename will be hard-coded to guarantee the correct order"
		return doc

	def _parse_args(self):
		args = dict()
		args["path"] = self.configs["path"]
		args["history"] = 50 # history window size
		args["step"] = 1 # steps in the future
		args["target"] = 1 # forcast window size
		self.args = args

	def _split(self, fold = 10, k = 9, val = False):
		# forward chaining split
		paths = self.args["path"]

		split_idx = int(len(paths) * k / fold) ############ unsafe here
		end_idx = split_idx + 1 + self.args["step"] + self.args["target"] #split_idx included
		if val:
			val_idx = int(end_idx*0.8)
			train_path = paths[:val_idx]
			val_path = paths[val_idx:end_idx]
			test_path = paths[end_idx:]
			return [train_path, test_path, val_path]
		else:
			train_path = paths[:end_idx]
			test_path = paths[end_idx:]
			return [train_path, test_path]

	def _pipeline(self, ds):
		def parse_file(file):
			ds = tf.data.TextLineDataset(file)
			ds = ds.skip(1) # skip the 1st line
			ds = ds.take(150) # 100
			def parse_line(line):
				val_str = tf.strings.split(line)
				val = tf.strings.to_number(val_str[1], out_type=tf.dtypes.float32)
				return val
			ds = ds.map(parse_line).batch(150)
			ds = ds.map(lambda x : tf.reshape(x, [150,])) # 100
			# ds = ds.map(lambda *lines : tf.stack(lines)) # this one only work for csv rows
			return ds # this is still a dataset


		dataset = tf.data.Dataset.from_tensor_slices(ds)
		dataset = dataset.map(parse_file)
		dataset = dataset.flat_map(lambda x: x) # now it's not dataset of dataset anymore
		dataset = dataset.map(self._normalize_simple)		
		def make_windows(dataset, shift = 1 , stride = 1):
			win_size = self.args["history"] + self.args["target"] + self.args["step"] - 1
			dataset = dataset.window(win_size, shift, stride)
			def subset_batch(subset):
				return subset.batch(win_size, drop_remainder=True)
			return dataset.flat_map(subset_batch)
		dataset = make_windows(dataset)
		def divide_train_label(batch):
			data = batch[:self.args["history"]]
			label = batch[-self.args["target"]:]
			return data, label
		dataset = dataset.map(divide_train_label)

		dataset = dataset.batch(10)
		dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return dataset

	@staticmethod
	def _normalize_simple(element):
		return (element-80)*0.01
		
		
	@staticmethod
	def _preprocess(element):
		pass



if __name__ == '__main__':
	# # configs = {"path" : features_path}
	# ds = ClassifierDataSet(configs)
	# dataset = ds("train")
	# for data, label in dataset:
	# 	tf.print(data, output_stream=sys.stdout)
	# 	tf.print(label, output_stream=sys.stdout)
	# # for item in dataset:
	# # 	import pdb; pdb.set_trace()
	# # 	tf.print(item[0], output_stream=sys.stdout)
	# # for feature, label in dataset.take(1):
	# # 	print(list(feature.keys()))
	# # 	print(feature['K1'])
	# # 	print(label)
	# print("Done")

	# df = pd.read_csv(test_single_file_path, sep='\s+')
	# df.to_csv(test, sep =',', index=False)
        
        import sys
        ds = SequentialDataset()
        ds.cache()
        d = ds()
        import pdb;pdb.set_trace()
        print(d.element_spec)
        for ele in d.take(10):
            # import pdb;pdb.set_trace()
            tf.print(ele, output_stream=sys.stdout)
