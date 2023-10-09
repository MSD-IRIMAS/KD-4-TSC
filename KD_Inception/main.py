import os
import numpy as np
import sklearn
import utils
from utils.utils import read_all_datasets, create_directory, get_best_teacher
from utils.constants import CLASSIFIERS, ARCHIVE_NAMES, ITERATIONS, ALPHALIST, TEMPERATURELIST, PATH_OUT, EPOCHS, depth

from training import Trainer


def callfit(out_dir, alpha = None, temperature = None):
    
    out_dir = out_dir + dataset_name + '/'
    if os.path.exists(out_dir + 'DONE'):
        print('Already computed')
    else:
        create_directory(out_dir)
        fit_classifier(out_dir, alpha, temperature)
        create_directory(out_dir + '/DONE')        


def fit_classifier(out_dir, alpha=None, temperature=None):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(clsf_name, input_shape, nb_classes, out_dir, best_teacher_model_path, alpha, temperature)
    classifier.fit_and_evaluate(x_train, y_train, x_test, y_test)


def create_classifier(clsf_name, input_shape, nb_classes, out_dir, best_teacher_model_path = None, alpha = None, temperature = None):    
    return Trainer(clsf_name, input_shape, nb_classes, out_dir, EPOCHS, depth, best_teacher_model_path, alpha, temperature)

############################################### main
# change this directory path in utils.constants for your machine
best_teacher_model_path, alpha, temperature = None, None, None
root_dir = PATH_OUT
path_teacher = PATH_OUT + '/results/teacher_rm/'

for archive_name in ARCHIVE_NAMES:
    
    datasets_dict = read_all_datasets()

    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
        for clsf_name in CLASSIFIERS:
            
            if clsf_name == 'student_rm':
                best_teacher_model_path  = get_best_teacher(dataset_name, path_teacher)

            for iter in range(ITERATIONS[clsf_name]):

                if clsf_name != 'student_rm':
                    out_dir = root_dir + '/results/'  + clsf_name + '/' + archive_name + '_itr_' + str(iter+1) + '/'  
                    callfit(out_dir)

                else: #student
                    for alpha in ALPHALIST:
                        for temperature in TEMPERATURELIST:
                            out_dir = root_dir + '/results/'  + clsf_name + '/alpha' + str(alpha) + '/temperature' + str(temperature)+  '/' + archive_name + '_itr_' + str(iter+1) + '/'
                            callfit(out_dir, alpha, temperature)