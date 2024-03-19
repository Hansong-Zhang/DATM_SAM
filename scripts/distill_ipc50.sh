


for d in "imagenette" "imagewoof" "imagemeow" "imagesquawk"
do
    for sel_prob in 0.1 0.3 0.5 0.7 0.9
    do
        python3 ../distill/DATM_SAM.py --cfg ../configs/ImageNet-Subsets/IPC50.yaml --subset $d --sel_p $sel_prob
    done
done
