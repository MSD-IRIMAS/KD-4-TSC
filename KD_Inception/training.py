import tensorflow as tf
from tensorflow import keras as keras
from utils.utils import create_inception_with_n_module, SaveBestModelDistiller, save_logs
from distiller import Distiller

import time

class Trainer:
    def __init__(self, model_name, input_shape, nb_classes, output_directory, epochs, depth, best_teacher_model_path=None, alpha=None, temperature=None):
        self.model_name = model_name
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.output_directory = output_directory
        self.epochs = epochs
        self.depth = depth
        self.alpha = alpha
        self.temperature = temperature
        self.best_teacher_model_path = best_teacher_model_path
        
        self.model = self.build_model()

    def build_model(self):
        if self.model_name == 'student_rm':
            best_teacher = create_inception_with_n_module(6, self.input_shape, self.nb_classes, "teacher_kd", nb_filters=32) 
            best_teacher.load_weights(self.best_teacher_model_path)
            best_teacher.summary()

            student = create_inception_with_n_module(self.depth, self.input_shape, self.nb_classes, "student", nb_filters=32) 
            student.summary()

            distiller = Distiller(student=student, teacher=best_teacher)
            distiller.compile(
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.CategoricalAccuracy()],
                student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
                distillation_loss_fn=keras.losses.KLDivergence(),
                alpha=self.alpha,
                temperature=self.temperature,
            )
            self.reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='student_loss', factor=0.5, patience=50,  min_lr=0.0001)
            file_path = self.output_directory + 'best_model.hdf5'
            self.model_checkpoint = SaveBestModelDistiller(distiller, file_path, self.epochs, save_best_metric='student_loss')

            return distiller

        elif self.model_name == 'studentAlone_rm':
            model_studentAlone = create_inception_with_n_module(self.depth, self.input_shape, self.nb_classes, model_name='studentAlone', nb_filters=32)
            model_studentAlone.summary()
            model_studentAlone.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
            self.reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
            file_path = self.output_directory + 'best_model.hdf5'
            self.model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

            return model_studentAlone
        
        elif self.model_name == 'teacher_rm':
            model_teacher = create_inception_with_n_module(self.depth, self.input_shape, self.nb_classes, model_name='teacher', nb_filters=32)
            model_teacher.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
            self.reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
            file_path = self.output_directory + 'best_model.hdf5'
            self.model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

            return model_teacher


    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=64, epochs=self.epochs, verbose=True, callbacks=[self.reduce_lr, self.model_checkpoint])       
        duration = time.time() - start_time
        if self.model_name == 'student_rm':
            self.model = self.model.student
            save_logs(self.model, self.output_directory, hist, self.epochs, x_test, y_test, duration, plot_test_acc=False, loss_column='student_loss')
        else:
            save_logs(self.model, self.output_directory, hist, self.epochs, x_test, y_test, duration, plot_test_acc=False)

        self.model.save(self.output_directory + 'last_model.hdf5')


        keras.backend.clear_session()
