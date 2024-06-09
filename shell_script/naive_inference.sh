channel=('RGB' 'RG' 'RB' 'BG' 'R' 'G' 'B')

for i in "${channel[@]}"; do 
    python inference.py --timestamp 06_08_16 --use_channel $i
done

