# [NeurIPS 2024] Towards Lossless Large-Scale Dataset Distillation via Foreground Separation

## Getting Started
**Step 1: Prepare the ImageNet-1k Dataset**
if you do not have it in your device, please download the training set by
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
```
and the validation set by
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
next, put the official [pytorch example script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) in the folder where you download the above two tar files, and run it by
```
bash extract_ILSVRC.sh
```
**Step 2: Create environment as follows**
```
conda env create -f environment.yaml
conda activate distillation
```
**Step 3: Generate expert trajectories** 
- Change the **data_path**(the folder of imagenet) and **buffer_path**(where to save the training trajectories) and **logs_dir**(where to save the buffer training logs) in buffer/buffer_FTD.py
- train the corresponding expert trajectories by
```
cd scripts
export CUDA_VISIBLE_DEVICES=xxx
bash run_buffer_xxxxxx.sh
```
**Step 4: Perform the distillation**

