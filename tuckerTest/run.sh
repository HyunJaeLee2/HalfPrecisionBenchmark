for i in 25 50 75; do

./test 416 3 3 16 ${i} ${i} &> profile/layer0_${i}.log
./test 208 16 3 32 ${i} ${i} &> profile/layer2_${i}.log 
./test 104 32 3 64 ${i} ${i} &> profile/layer4_${i}.log
./test 52 64 3 128 ${i} ${i} &> profile/layer6_${i}.log
./test 26 128 3 256 ${i} ${i} &> profile/layer8_${i}.log
./test 13 256 3 512 ${i} ${i} &> profile/layer10_${i}.log
./test 13 512 3 1024 ${i} ${i} &> profile/layer12_${i}.log
./test 13 1024 3 1024 ${i} ${i} &> profile/layer13_${i}.log

done
