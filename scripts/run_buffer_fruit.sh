






python3 ../buffer/buffer_FTD.py --dataset=ImageNet --subset=imagefruit --model=ConvNetD5  \
                                --num_experts=100                                         \
                                --lr_teacher=0.01                                         \
                                --batch_train=128                                         \
                                --batch_real=128                                          \
                                --dsa=True                                                \
                                --dsa_strategy=color_crop_cutout_flip_scale_rotate        \
                                --train_epochs=60                                         \
                                --mom=0 --l2=0                                                    \
                                --save_interval=10                                        \
                                --rho_max=0.01 --rho_min=0.01 --alpha=0.3  \
                                --adaptive=True \
                                --data_path=/home/wangkai/big_space/datasets/imagenet/ \
                                --buffer_path=/home/wangkai/big_space/hs_zhang/Z_demo/DATM/buffer_storage/ \
                                --logs_dir=/home/wangkai/big_space/hs_zhang/Z_demo/DATM/buffer_logs/
