# Deep Learning-Based Radiomics for Prognostic Stratification of Low-Grade Gliomas Using a Multiple-Gene Signature

To our knowledge, no previous study has applied a deep learning-based radiomic approach for novel biomarkers, such as the aforementioned multiple-gene genetic sig-natures. Our study utilized the 3-gene signature that stratifies LGG patients into high- and low-risk groups, according to Xiao et al. (doi:10.7717/peerj.8312). We aimed to determine whether neuroimaging features on preoperative MRI studies entered into a CNN pipeline could facilitate the proposed LGG stratification. The code was developed based on the project by Zunair et. al (https://keras.io/examples/vision/3D_image_classification).

## Files

The .py files are sorted by the recommended order of running.

1) get_clinic.py: This file serves as getting clinical information and creating data for risk groups.
2) get_nifti.py: This file serves as reading the imaging data.
3) train.py: This file serves to train the models.
4) predict.py: This file serves to make predictions from trained models and get performance metrics.

augment.py, configs.py and generator.py should be in the working directory when running these .py files.

## Authors of the Study

1) Mert Karabacak
Department of Neurosurgery, Mount Sinai Health System, 1468 Madison Avenue, New York, NY, 10029, USA; mert.karabacak@mountsinai.org
2) Burak B. Ozkara
Department of Neuroradiology, MD Anderson Cancer Center, 1400 Pressler Street, Houston, TX, 77030, USA; bbozkara@mdanderson.org
3) Kaan Senparlak
School of Computation, Information and Technology, Technical University of Munich, Theresienstr. 90, München, Bayern, 80333, Germany; kaansenparlak@hotmail.com
4) Sotirios Bisdas*
Department of Neuroradiology, The National Hospital for Neurology and Neurosurgery, University College London NHS Foundation Trust, London WC1N 3BG, UK; s.bisdas@ucl.ac.uk
