Data preparation: 
https://www.kaggle.com/code/adityakumarsinha/data-creation-v1/notebook
or data-creation-v1.ipynb

This notebook requires librosa 0.9.2 for resampling of tdcsfog data.
This kernel will create numpy data in following folders:
tdcsfog data will be created in  tdcsfog_np folder. defog data will be created in defog_np.
For each series two files will be created.
file with name <id>_sig.npy will contain series value for columns ([ 'AccV', 'AccML', 'AccAP'])
file with name <id>_tgt will contain target values ['StartHesitation', 'Turn', 'Walking']

After data creation, for training code in following notebook should be used:
https://www.kaggle.com/code/adityakumarsinha/wavenet-4096-v6/notebook
or wavenet-4096-v6.ipynb

Note: It needs torch 1.12 and will not finish in kaggle allocated time
#!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

The notebook is using kaggle kernel output above notebook (https://www.kaggle.com/code/adityakumarsinha/data-creation-v1/notebook)
Otherwise INPUT_PATH_NP variable needs to point to folder that contains defog_np and tdcsfog_np folders.
Other variable INPUT_PATH should point to competition data. In kaggle kernels it is available at  '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction'

The code will create folder trained-models-wavenet_4096 -v6. This folder contains checkpoint for different folds with name wavenet_4096-fold{fold_num}_{epoch}
At the end of training pick last (it is checkpoint fo epoch with best average precision score) checkpoints for each fold. Use the checkpoints in inference notebook which is available at path
https://www.kaggle.com/code/adityakumarsinha/wavenet-subm-focal-v1-1/notebook

The code expects that inference data should be available at /kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test within subfolder tdcsfog and defog
The checkpoints need to be specified in ckpt_path variable. For example.
ckpt_paths = ['/kaggle/input/gait-wavenet-focal/wavenet_4096-fold0_18.pth',
            '/kaggle/input/gait-wavenet-focal/wavenet_4096-fold1_54.pth',
            '/kaggle/input/gait-wavenet-focal/wavenet_4096-fold2_19.pth',
            '/kaggle/input/gait-wavenet-focal/wavenet_4096-fold3_21.pth',
            '/kaggle/input/gait-wavenet-focal/wavenet_4096-fold4_45.pth'
            ]

These steps will reproduce one of the models I have used in final submission, but it has the best score.
You can also refer https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/418275


Pre-training with unlabeled data: 
See the notebook to generate input data for pre-training
pq_files = glob.glob(f'{INPUT_DIR}/unlabeled/*.parquet')
https://www.kaggle.com/code/adityakumarsinha/unlabeled-data-creation/notebook

unlabeled/ Folder containing the unannotated data series from the daily dataset, one series per subject. Forty-five of the subjects also have series in the defog dataset, some in the training split and some in the test split. Accelerometer data has units of g. 

