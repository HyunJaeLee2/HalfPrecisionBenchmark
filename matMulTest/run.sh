nvprof --events all ./test $1 half &> half_${1}.nvprof 
nvprof --events all ./test $1 normal &> normal_${1}.nvprof
