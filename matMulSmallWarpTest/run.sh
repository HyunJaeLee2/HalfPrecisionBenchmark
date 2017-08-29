FOLDER=${1}x${1}
mkdir profile/${FOLDER}
for i in 1 2 4 8 16; do
    nvprof --events all ./test $1 $i half &> profile/${FOLDER}/half_${1}_${i}.nvprof
    nvprof --events all ./test $1 $i normal &> profile/${FOLDER}/normal_${1}_${i}.nvprof
done
