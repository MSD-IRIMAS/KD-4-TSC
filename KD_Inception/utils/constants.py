UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF',
                            'Chinatown', 'ChlorineConcentration', 'CinCECGtorso', 'Coffee',
                            'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                            'EOGVerticalSignal','EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'FISH',
                            'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan',
                            'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 
                            'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 
                            'InsectWingbeatSound', 'ItalyPowerDemand','LargeKitchenAppliances', 'Lightning2', 'Lightning7', 
                            'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 
                            'MiddlePhalanxTW', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 
                            'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 
                            'PigAirwayPressure','PigArtPressure','PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock',
                            'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2','SemgHandSubjectCh2', 'ShapeletSim', 
                            'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                            'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'uWaveGestureLibraryX', 'uWaveGestureLibraryY',
                            'uWaveGestureLibraryZ', 'wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'yoga']

UNIVARIATE_DATASET_NAMES_2018 = ['ArrowHead', 'ACSF1', 'Adiac', 'Beef',]
UNIVARIATE_DATASET_NAMES_2018 = ['ArrowHead',]

ARCHIVE_NAMES = ['UCRArchive_2018']
dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}

CLASSIFIERS = ['teacher_rm', 'student_rm', 'studentAlone_rm']
CLASSIFIERS = ['studentAlone_rm']
ITERATIONS = {
              'teacher_rm': 5,  # nb of random runs for random initializations
              'student_rm': 5,  # nb of random runs for random initializations
              'studentAlone_rm': 5  # nb of random runs for random initializations
             }
            
# CLASSIFIERS = ['StudentAlone']
# CLASSIFIERS = ['Student']

EPOCHS = 1500
depth = 4

BEST_TEACHER_ONLY = True
# If true, be sure to have copied the best teacher results in best_teacher folder using
# the script 'copy_best_teacher.py'

ALPHALIST = [0.3]
TEMPERATURELIST = [10,]


PATH_DATA = '/home/javidan/Codes/UCRArchive_2018'
PATH_OUT = '/home/javidan/Codes/Results'