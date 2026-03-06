import os

if os.path.exists('/kaggle/input'):
    # Kaggle environment
    DATA_ROOT = '/kaggle/input/datasets/divyamagg/l2-arctic-data/'
    LABEL_ROOT = '/kaggle/input/datasets/hongvitchugo/l2-arctic-labels/'
    # Labels-folder CSVs already contain the .wav extension in the Path column
    WAV_SUFFIX = ''
else:
    # Local environment — original mdd_asv_baseline CSVs do not include .wav extension
    DATA_ROOT = '../EN_MDD/WAV/'
    LABEL_ROOT = './'
    WAV_SUFFIX = '.wav'

CHECKPOINT_DIR = './checkpoint'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
