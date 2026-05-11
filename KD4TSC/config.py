"""Configuration file for Knowledge Distillation experiments."""

import os
import numpy as np
# Dataset configuration
UNIVARIATE_DATASET_NAMES_2018 = [
    'ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF',
    'Chinatown', 'ChlorineConcentration', 'CinCECGtorso', 'Coffee',
    'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
    'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
    'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'FISH',
    'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan',
    'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics',    
    'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain',
    'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
    'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
    'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock',
    'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim',
    'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
    'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
    'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
]

UNIVARIATE_DATASET_NAMES_2018 = [ 'ACSF1' ]

ARCHIVE_NAMES = ['UCRArchive_2018']

DATASET_NAMES_FOR_ARCHIVE = {
    'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018
}

# Model types
# CLASSIFIERS = ['teacher', 'student_kd', 'student_alone']
CLASSIFIERS = ['student_kd']  

ITERATIONS = {
    'teacher': 5,
    'student_kd': 5,
    'student_alone': 5
}

# Training hyperparameters
EPOCHS = 1500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 50
MIN_LR = 0.0001
LR_FACTOR = 0.5

# ============================================================================
# Architecture Selection
# ============================================================================
ARCHITECTURE = 'convtran'

# ============================================================================
# INCEPTION Model hyperparameters
# ============================================================================
INCEPTION_TEACHER_DEPTH = 6
INCEPTION_STUDENT_DEPTH = 4
INCEPTION_NB_FILTERS = 32
INCEPTION_BOTTLENECK_SIZE = 32
INCEPTION_KERNEL_SIZE = 40


# ============================================================================
# FCN Model hyperparameters  
# ============================================================================
# FCN supports TWO compression dimensions:
# 1. DEPTH: Number of convolutional layers (length of filters list)
# 2. WIDTH: Number of filters per layer (values in filters list)

# Teacher FCN - Standard depth (3 layers)
FCN_TEACHER_FILTERS = [128, 256, 128]  # 3 layers
FCN_TEACHER_KERNEL_SIZES = [8, 5, 3]   # Optional: kernel size per layer

# ============================================================================
# Student FCN Configurations - Mix depth AND width compression!
# ============================================================================

# Option 1: WIDTH COMPRESSION ONLY (same depth, fewer filters)
# Recommended for most cases
FCN_STUDENT_FILTERS = [20, 40, 20]    # 3 layers, 50% filters
# FCN_STUDENT_FILTERS = [32, 64, 32]   # 3 layers, 75% filters

# Option 2: DEPTH COMPRESSION ONLY (fewer layers, same filters)  
# Simpler model, faster inference
# FCN_STUDENT_FILTERS = [128, 256]     # 2 layers, full filters
# FCN_STUDENT_FILTERS = [128]          # 1 layer, full filters

# Option 3: DEPTH + WIDTH COMPRESSION (fewer layers AND fewer filters)
# Maximum compression!
# FCN_STUDENT_FILTERS = [64, 128]      # 2 layers, 50% filters
# FCN_STUDENT_FILTERS = [32, 64]       # 2 layers, 75% filters

# Option 4: DEEPER STUDENT (more layers, fewer filters per layer)
# More expressive, still compressed
# FCN_STUDENT_FILTERS = [64, 96, 128, 96, 64]  # 5 layers, varied filters

# Custom kernel sizes (optional, defaults to [8, 5, 3, 3, ...])
FCN_STUDENT_KERNEL_SIZES = None  # Use default or specify: [8, 5, 3]


# ============================================================================
# ConvTran Model hyperparameters  
# ============================================================================
TEACHER_NUM_HEADS = 8
STUDENT_NUM_HEADS = 6

# Backward compatibility aliases (for Inception)
TEACHER_DEPTH = INCEPTION_TEACHER_DEPTH
STUDENT_DEPTH = INCEPTION_STUDENT_DEPTH
NB_FILTERS = INCEPTION_NB_FILTERS
BOTTLENECK_SIZE = INCEPTION_BOTTLENECK_SIZE
KERNEL_SIZE = INCEPTION_KERNEL_SIZE


# Knowledge Distillation hyperparameters
ALPHA_LIST = [0.5 ]  # Weight for student loss vs distillation loss
TEMPERATURE_LIST = [10]  # Temperature for softening predictions

# Paths (modify these for your system)
PATH_DATA = '/home/jabdullayev/phd/datasets/UCRArchive_2018/'
PATH_OUT = os.path.join('/home/jabdullayev/phd/projects/KD4TSC/Results/', ARCHITECTURE.upper())


# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Best teacher configuration
BEST_TEACHER_ONLY = True
PATH_TEACHER = os.path.join(PATH_OUT, 'teacher')