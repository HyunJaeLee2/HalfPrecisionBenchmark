./test 416 3 3 16 50 50 float &> profile/layer1_float.log
./test 208 16 3 32 50 50 float &> profile/layer3_float.log 
./test 104 32 3 64 50 50 float &> profile/layer5_float.log
./test 52 64 3 128 50 50 float &> profile/layer7_float.log
./test 26 128 3 256 50 50 float &> profile/layer9_float.log
./test 13 256 3 512 50 50 float &> profile/layer11_float.log
./test 13 512 3 1024 50 50 float &> profile/layer13_float.log

./test 416 3 3 16 50 50 half &> profile/layer1_half.log
./test 208 16 3 32 50 50 half &> profile/layer3_half.log 
./test 104 32 3 64 50 50 half &> profile/layer5_half.log
./test 52 64 3 128 50 50 half &> profile/layer7_half.log
./test 26 128 3 256 50 50 half &> profile/layer9_half.log
./test 13 256 3 512 50 50 half &> profile/layer11_half.log
./test 13 512 3 1024 50 50 half &> profile/layer13_half.log
