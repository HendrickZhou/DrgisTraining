import os
import sys
script_path = os.path.abspath('')
project_path = script_path[:script_path.rfind('DrgisTraining')]+ "DrgisTraining/"
data_path = project_path + 'data/' 

travel_path_f = lambda i : data_path + "Open_Travel/" + "CB550_open_travel_t%d.dat"%(i)

# checkpts and tf summary path for tensorflow model
default_train_sum_path = project_path + 'temp/'
checkpoint_rnn_dir = default_train_sum_path + "checkpoint_rnn/"
tensorboard_rnn_path = default_train_sum_path + "summary_rnn/"
checkpoint_rnn_dir_n = lambda n : default_train_sum_path + "checkpoint_rnn_%d"%(n)
tensorboard_rnn_dir_n = lambda n : default_train_sum_path + "summary_rnn_%d"%(n)
checkpoint_train_path = "train-{epoch:03d}.hdf5"
checkpoint_val_path = "val-{epoch:03d}.hdf5"

model_predictor_dir = default_train_sum_path + "model_rnn/"
model_predictor_dir_n = lambda n : default_train_sum_path + "model%d_rnn/"%(n)
