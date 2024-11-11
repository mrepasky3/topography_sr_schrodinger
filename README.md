# Prerequisites

### Download Network Checkpoints
The variational autoencoder (VAE) and Schrödinger bridge (SB) network checkpoints are required to perform super-resolution.
1) Download [polar_vae_20MPP_200.zip](https://drive.google.com/file/d/1gMKrFVpaQt28Z0Dl7d1QQ6xWIxdpPgW2/view?usp=sharing) and [bettersun_bridge_20-320MPP_005.zip](https://drive.google.com/file/d/1VYjbiLg32NOGN2MBrZOTjtZ18UZuiSdr/view?usp=sharing) from the Google Drive link
2) Unzip `polar_vae_20MPP_200.zip` into `PATH_TO_SR_FOLDER/results/`
3) Unzip `bettersun_bridge_20-320MPP_005.zip` into `PATH_TO_SR_FOLDER/results/`

### Install Python Dependencies
1) `cd PATH_TO_SR_FOLDER`
2) `pip install -r requirements.txt`

To re-create the training, validation, and testing datasets, see the following sub-sections.

### ShadowSpy
ShadowSpy, a ray-tracing DEM rendering software, is required to create datasets for training, validation, and testing. Follow the instructions at https://github.com/steo85it/shadowspy/tree/v0.1.0 to install ShadowSpy version 0.1.0.

### LOLA DEM
The topography model used to train this model can be found at https://pgda.gsfc.nasa.gov/products/90, the file `LDEM_80S_20MPP_ADJ.TIF`. This is the processed LOLA DEM covering the lunar south polar 80&deg;-90&deg; south at 20 meters-per-pixel.


# Construct Datasets
The datasets consist of processed LOLA DEM patches which are each rendered from 360 illumination angles.

The first stage of dataset generation creates sub-directories for each patch, including the patch TIF and a TIF containing the 360 images. If each patch is 96 pixels, this corresponds to 315 "columns" in the LOLA DEM (`COLUMN_ID={0, ..., 314}`) each yielding 315 patches of size 96 by 96 pixels. We rendering low-elevation sun and high-elevation sun separately `ILLUM_TYPE={lowsunele, highsunele}`.
1) `cd PATH_TO_SR_FOLDER/process_data/`
2) `python illuminate_patches.py --tif_path=PATH_TO_LOLA_DEM --illum_type=ILLUM_TYPE --column=COLUMN_ID`

The previous step will result in a directory `south_ILLUM_TYPE_96px_20MPP/patch_XXXXXX` containing files `hd_patch.tif` and `imgs_patch.tif`, which include a DEM patch and the associated images. Once a large number of these patches are rendered, the following steps can be used to aggregate them into hdf5 files.
1) `cd PATH_TO_SR_FOLDER/process_data/`
2) `python convert_hdf5.py --illum_type=ILLUM_TYPE --patch_start=START_IDX --patch_end=END_IDX`

The `START_IDX` and `END_IDX` control how many patches are contained within this particular hdf5 file. For training, we have distributed the 99,225 patches at each pole between 99 files for each illumination type (high sun elevations versus low sun elevations).

We provide the train/val/test indices for the entire dataset at `PATH_TO_SR_FOLDER/process_data/thresh2242_{train,val,test}_idx.npy`. If you do not create the entire dataset, replace these with your own partitioning.


# Model Training and Evaluation
The proposed model is a latent-space Schrödinger bridge. Therefore, VAE model training is required first, followed by SB model training. During training, model checkpoints (containing the model weights) will be saved intermittently. Additionally, validation metrics will be computed on the validation split of the data, and plots of the validation metrics and sampled DEM patches will be provided.

### Variational Autoencoder (VAE) Training
Train a VAE, providing a model identifier integer `VAE_ID`

`python autoencoder_trainer.py --model_id=VAE_ID`

### Schrödinger Bridge Training
The SB requires a pre-trained VAE, with model id `VAE_ID` and model checkpoint number `VAE_CKPT`.

`python trainer.py --model_id=MODEL_ID --encoder_id=VAE_ID --encoder_iter=VAE_CKPT --downsample=16 --num_imgs_lower=5 --num_imgs_upper=100`

### Schrödinger Bridge Evaluation
Evaluation metrics on the test dataset can be computed for a specific model checkpoint/epoch.

`python evaluator.py --model_id=MODEL_ID --model_epoch=MODEL_EPOCH --num_imgs=NUM_IMGS --lat_lower=LAT_LOWER --lat_upper=LAT_UPPER --nfe=NUM_FUNCTION_EVALS --n_repeat=N_REPEAT`

Parameter `NUM_IMGS` controls the number of images to provide to the model for evaluation. `LAT_LOWER` and `LAT_UPPER` control the latitude bounds of the DEM patches, which controls the illumination conditions. `NUM_FUNCTION_EVALS` corresponds to the number of discretization steps to take for sampling (higher NFE tends to produce better samples). `N_REPEAT` represents the number of clones per patch - the metrics are computed with the average prediction for each patch.


# Topography Super-Resolution
We provide scripts to conduct 16x topography super-resolution on 96 by 96 pixel patches, a large region, and on user-provided DEMs with imagery.

### Small Patch Super-Resolution
A pre-trained model can be used to conduct super-resolution of individual patches from the rendered dataset.

`python patch_superres.py --model_id=MODEL_ID --model_epoch=MODEL_EPOCH --num_imgs=NUM_IMGS --lon=LONGITUDE --lat=LATITUDE --nfe=NUM_FUNCTION_EVALS --n_repeat=N_REPEAT --split=SPLIT --patch_idx=PATCH_ID`

`LONGITUDE` and `LATITUDE` specify the assumed location of the patch on the lunar surface (which controls the illumination conditions). `PATCH_ID` specifies the patch index in the `SPLIT` partition of the dataset on which to conduct super-resolution.

### Large Region Super-Resolution
We provide a rendered large DEM patch at a Google Drive link. These files can be placed in the folder `PATH_TO_SR_FOLDER/dem/`. A pre-trained model can be applied for super-resolution of this large region.

`python region_superres.py --model_id=MODEL_ID --model_epoch=MODEL_EPOCH --num_imgs=NUM_IMGS --lon=LONGITUDE --lat=LATITUDE --nfe=NUM_FUNCTION_EVALS --n_repeat=N_REPEAT --region_size=REGION_SIZE --stride=48 --weight_sigma=2`

The `REGION_SIZE` in pixels parameter constrains the size of the large region (up to 3840 pixels). The `stride` controls the "step" of the moving window (in pixels) and `weight_sigma` is the weight of the Gaussian convolution to the grassfire filter.