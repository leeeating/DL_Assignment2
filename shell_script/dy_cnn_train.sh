channel=('RGB' 'RG' 'RB' 'BG' 'R' 'G' 'B')

for i in "${channel[@]}"; do 
    python training.py --timestamp 06_08_16 --model_name dy_cnn --use_channel $i
done
 
