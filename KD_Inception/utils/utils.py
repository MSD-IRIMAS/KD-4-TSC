import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import PATH_DATA
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras as keras

def read_all_datasets(split_val=False):
    datasets_dict = {}
    cur_root_dir = PATH_DATA
    
    
    for dataset_name in DATASET_NAMES_2018:
        root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'
    
        df_train = pd.read_csv(root_dir_dataset + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
        df_test = pd.read_csv(root_dir_dataset + dataset_name + '_TEST.tsv', sep='\t', header=None)   

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]
        
        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])
        
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])
        
        x_train = x_train.values
        x_test = x_test.values
        
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    
    
    return datasets_dict

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def read_dataset(root_dir, dataset_name):
    datasets_dict = {}
    cur_root_dir = PATH_DATA
    root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'

    df_train = pd.read_csv(root_dir_dataset + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    df_test = pd.read_csv(root_dir_dataset + dataset_name + '_TEST.tsv', sep='\t', header=None)
    
    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]
    
    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])
    
    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])
    
    x_train = x_train.values
    x_test = x_test.values
    
    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    
    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())
    
    return datasets_dict

def get_best_teacher(dataset_name, path_teacher):
    teacher_folders = ['UCRArchive_2018_itr_1', 'UCRArchive_2018_itr_2', 'UCRArchive_2018_itr_3', 'UCRArchive_2018_itr_4', 'UCRArchive_2018_itr_5']
    train_losses = []
    val_accuracies = []

    for sup_name in teacher_folders:
        teacher_metrics_df = pd.read_csv(path_teacher + sup_name + '/' + dataset_name + '/' + 'df_best_model.csv')
        teacher_train_loss = teacher_metrics_df['best_model_train_loss'][0]
        train_losses.append(teacher_train_loss)
        teacher_metrics_df = pd.read_csv(path_teacher + sup_name + '/' + dataset_name + '/' + 'df_metrics.csv')
        teacher_val_acc = teacher_metrics_df['accuracy'][0]
        val_accuracies.append(teacher_val_acc)

        min_loss_index = train_losses.index(min(train_losses))
        best_teacher_model_path = path_teacher + teacher_folders[min_loss_index] + '/' + dataset_name + '/' + 'best_model.hdf5'
    
    return  best_teacher_model_path

def inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, bottleneck_size=32, nb_filters=32, kernel_size=40):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                    padding='same', activation=activation, use_bias=False)(max_pool_1)


    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def create_inception_with_n_module(depth, input_shape, nb_classes, model_name, nb_filters):
    
    input_layer = keras.layers.Input(input_shape)
    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = inception_module(x, nb_filters=nb_filters)

        if d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    if model_name == 'teacher' or model_name == 'studentAlone':
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    else:
        output_layer = keras.layers.Dense(nb_classes)(gap_layer)

    new_model = keras.models.Model(inputs=input_layer, outputs=output_layer, name=model_name)   

    return new_model

# def create_inception_with_n_module(depth, input_shape, nb_classes, model_name, nb_filters):
    
#     input_layer = keras.layers.Input(input_shape)
#     x = input_layer
#     input_res = input_layer

#     for d in range(depth):
#         x = inception_module(x, nb_filters=nb_filters)
#         if d % 3 == 2:
#             x = shortcut_layer(input_res, x)
#             input_res = x

#     gap_layer = keras.layers.GlobalAveragePooling1D()(x)

#     if model_name == 'student' or model_name == 'teacher_kd':
#         output_layer = keras.layers.Dense(nb_classes)(gap_layer)
#     else:
#         output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

#     model = keras.models.Model(inputs=input_layer, outputs=output_layer, name=model_name)   

#     return model

    return new_model
# -----------------------------------------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def save_logs(model, output_directory, hist, epochs, x_test, y_test, duration, plot_test_acc=True, loss_column='loss'):
    
    model.load_weights(output_directory + 'best_model.hdf5')
    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(x_test, batch_size=64)
    y_pred = np.argmax(y_pred, axis=1)

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)
   
    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    if loss_column == 'student_loss':
        index_best_model = hist_df.loc[int(epochs*(2/3)):]['student_loss'].idxmin()
    else:
        index_best_model = hist_df['loss'].idxmin()
    
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model[loss_column]
    df_best_model['best_model_train_acc'] = row_best_model['categorical_accuracy']
    df_best_model['best_model_nb_epoch'] = index_best_model
    
    if plot_test_acc:
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
        df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

class SaveBestModelDistiller(tf.keras.callbacks.Callback):
    def __init__(self, model, file_path, epochs, save_best_metric='loss'):
        self.save_best_metric = save_best_metric
        self.best = float('inf')
        self.model = model
        self.file_path = file_path
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]               
        if metric_value < self.best and epoch > int(self.epochs * (2/3)):
            self.best = metric_value
            self.model.student.save_weights(self.file_path)
