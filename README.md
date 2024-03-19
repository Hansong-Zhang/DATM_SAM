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
**Step 2: Download the Pre-trained Segment Anything Model Parameters**
find a folder that you would like to hold the parameters of Segment Anything Model (need 2.4 GB)
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
**Step 3: Create environment as follows**
```
conda env create -f environment.yaml
conda activate distill_sam
```
**Step 4: Generate expert trajectories** 
- Change the **data_path** (the folder of imagenet) and **buffer_path** (where to save the training trajectories) and **logs_dir** (where to save the buffer training logs) in buffer/buffer_FTD.py
- train the corresponding expert trajectories by
```
cd scripts
export CUDA_VISIBLE_DEVICES=xxx
bash run_buffer_xxxxxx.sh
```
**Step 5: Perform the distillation**
- first, you need to change the paths in utils/cfg.py
1. **buffer_path**: the path to the expert trajectories in **Step 3**
2. **data_path**: the path to hold imagenet dataset
3. **save_dir**: the path to save the synthetic images, visualizations, and logs
4. **sam_path**: the path to the downloaded *Segment Anything Model*
- second, you need to change which gpu to run the experiment on, change the **device** parameter in *configs/ImageNet-Subsets/IPCxxxxxx.yaml*
1. for parallel processing, you can set multiple numbers in **device**, such as *device: [0,1,2,3]*
- Third, run the bash file in scripts
```
cd scripts
bash distill_ipcxxxxxx.sh
```














