#CLEAR=~/clear_cache.sh
CLEAR=
TINYYOLO=tiny-yolo
SSD=ssd

if [ $1 -eq 1 ];
then 
    for i in 25 50 75; do
        ${CLEAR}
        ./test 416 3 3 16 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer0_${i}.log
        ${CLEAR}
        ./test 208 16 3 32 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer2_${i}.log 
        ${CLEAR}
        ./test 104 32 3 64 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer4_${i}.log
        ${CLEAR}
        ./test 52 64 3 128 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer6_${i}.log
        ${CLEAR}
        ./test 26 128 3 256 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer8_${i}.log
        ${CLEAR}
        ./test 13 256 3 512 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer10_${i}.log
        ${CLEAR}
        ./test 13 512 3 1024 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer12_${i}.log
        ${CLEAR}
        ./test 13 1024 3 1024 1 1 ${i} ${i} &> profile/${TINYYOLO}/layer13_${i}.log
    done
elif [ $1 -eq 2 ]; then
    for i in 25 50 75; do
        ${CLEAR}
        ./test 19 1024 1 256 0 1 ${i} ${i} &> profile/${SSD}/conv6_1_${i}.log
        ${CLEAR}
        ./test 19 256 3 512 1 2 ${i} ${i} &> profile/${SSD}/conv6_2_${i}.log 
        ${CLEAR}
        ./test 10 512 1 128 0 1 ${i} ${i} &> profile/${SSD}/conv7_1_${i}.log
        ${CLEAR}
        ./test 10 128 3 256 1 2 ${i} ${i} &> profile/${SSD}/conv7_2_${i}.log
        ${CLEAR}
        ./test 5 256 1 128 0 1 ${i} ${i} &> profile/${SSD}/conv8_1_${i}.log
        ${CLEAR}
        ./test 5 128 3 256 0 1 ${i} ${i} &> profile/${SSD}/conv8_2_${i}.log
        ${CLEAR}
        ./test 3 256 1 128 0 1 ${i} ${i} &> profile/${SSD}/conv9_1_${i}.log
        ${CLEAR}
        ./test 3 128 3 256 0 1 ${i} ${i} &> profile/${SSD}/conv9_2_${i}.log
    done
fi
