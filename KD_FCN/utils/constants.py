UNIVARIATE_DATASET_NAMES_2018 =  ['ArrowHead', 'Wine', 'FreezerSmallTrain', 'OliveOil',  'FordB', 'Car', 'NonInvasiveFetalECGThorax2', 
                                 'TwoPatterns', 'InsectWingbeatSound', 'BeetleFly', 'Yoga', 'InlineSkate', 'FaceAll',
                                 'EOGVerticalSignal', 'Ham', 'MoteStrain', 'ProximalPhalanxTW', 'WordSynonyms', 'Lightning7', 
                                 'GunPointOldVersusYoung', 'Earthquakes']

CLASSIFIERS = ['teacher']
#CLASSIFIERS = ['Student', 'StudentAlone']

ARCHIVE_NAMES = ['UCRArchive_2018']

dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}
EPOCHS = 2000

ITERATIONS = 5  # nb of random runs for random initializations
ITERATIONS_STUDENT_ALONE = 5 # nb of random runs for random initializations
ITERATIONS_STUDENT = 5

BEST_TEACHER_ONLY = True
# If true, be sure to have copied the best teacher results in best_teacher folder using
# the script 'copy_best_teacher.py'

#ALPHALIST = [i/100 for i in range(0, 101, 1)]
ALPHALIST = [0.1]
TEMPERATURELIST = [10]
PATH_DATA = "/home/jabdullayev/phd/datasets/UCRArchive_2018"
PATH_OUT = "."

FILTERS = [64, 32, 28, 24, 20, 16, 12, 8, 4]   
FILTERS2 = [128, 64, 56, 48, 40, 32, 24, 16, 8]

LAYERS = 3
SEPARABLE_CONV = False
