# [NeurIPS 2024] Towards Lossless Large-Scale Dataset Distillation via Foreground Separation

## Getting Started
1. Create environment as follows
```
conda env create -f environment.yaml
conda activate distillation
```
2. Generate expert trajectories
- **Step 1: Change the data path and buffer path in buffer/buffer_FTD.py**
```
cd buffer
python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --train_epochs=100 --num_experts=100 --zca --buffer_path=../buffer_storage/ --data_path=../dataset/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --lr_teacher=0.01 --mom=0. --batch_train=256
```
3. Perform the distillation
```
cd distill
python DATM.py --cfg ../configs/xxxx.yaml
```
## Evaluation
We provide a simple script for evaluating the distilled datasets.
```
cd distill
python evaluation.py --lr_dir=path_to_lr --data_dir=path_to_images --label_dir=path_to_labels --zca
```
## Acknowledgement
Our code is built upon [MTT](https://github.com/GeorgeCazenavette/mtt-distillation) and [FTD](https://github.com/AngusDujw/FTD-distillation).
## Citation
If you find our code useful for your research, please cite our paper.
```
@inproceedings{guo2024lossless,
      title={Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching}, 
      author={Ziyao Guo and Kai Wang and George Cazenavette and Hui Li and Kaipeng Zhang and Yang You},
      year={2024},
      booktitle={The Twelfth International Conference on Learning Representations}
}
```
