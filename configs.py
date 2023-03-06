from enum import Enum


class MRI(Enum):
    T1 = 0
    T2 = 1
    T1C = 2
    FLAIR = 3


mri_str = ["t1", "t2", "t1c", "flair"]

### Configurations ###
mri_types = [MRI.T1, MRI.T2]
width = 256
height = 256
depth = 20
channel = len(mri_types)
val_set_size = 20
cnn_solo = True
mlp_solo = False
classification = True
dim = 12
batch_size = 2
epochs = 100
