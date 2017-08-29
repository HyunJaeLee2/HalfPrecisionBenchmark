#FOLDER=${i}x${i}
#mkdir profile/${FOLDER}
for i in 32 64 128 256 512 1024 2048 4096; do
    echo "hihihi ${i}"
    ./test $i $i $i normal
done

#nvprof --events all tex0_cache_sector_queries tex1_cache_sector_queries tex0_cache_sector_misses tex1_cache_sector_misses elapsed_cycles_sm warps_launched inst_issued0 inst_issued1 inst_issued2 inst_executed fb_subp0_read_sectors fb_subp1_read_sectors fb_subp0_write_sectors fb_subp1_write_sectors
