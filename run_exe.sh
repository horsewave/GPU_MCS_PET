#rm ./gpu_mcs_eclipse
#rm /data/PET/mr_pet_temp/Ma/software/data/random_OFOV/Random-OFOV-activity-C11-2016-09-08/use_dead_scat_correction/XB1BN286N-BI-09/scatterMCS/*
#read -p "Press enter to continue"
#./gpu_mcs_eclipse
#wait 
#vi /data/PET/mr_pet_temp/Ma/software/data/random_OFOV/Random-OFOV-activity-C11-2016-09-08/use_dead_scat_correction/XB1BN286N-BI-09/scatterMCS/XB1BN286N-BI-09_0-1222_gpuSimu_0_log






rm ./gpu_mcs_eclipse
rm /data/PET/mr_pet_temp/Ma/software/data/gpupet/ofov/test/scatterMCS/*
read -p "Press enter to continue"
./gpu_mcs_eclipse
wait 
vi /data/PET/mr_pet_temp/Ma/software/data/gpupet/ofov/test/scatterMCS/test_gpuSimu_0_log









#brain phanto
#rm ./gpu_mcs_eclipse
#rm /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/Brain_phantom_ma/gpu_mcs/scatterMCS/*
#read -p "Press enter to continue"
#./gpu_mcs_eclipse
#wait 
#vi /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/Brain_phantom_ma/gpu_mcs/scatterMCS/brain_phantom_gpuSimu_0_log

#2 sphere phantom

#rm ./gpu_mcs_eclipse
#rm /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/XB1BN305N-BI/XB1BN305N-BI-01/scatterMCS/XB1BN305N-BI-01_0-1800_gpu*
#read -p "Press enter to continue"
#./gpu_mcs_eclipse
#wait 
#vi /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/XB1BN305N-BI/XB1BN305N-BI-01/scatterMCS/XB1BN305N-BI-01_0-1800_gpuSimu_0_log



#3: cubic phantom
#LOG_PATH="/data/PET/mr_pet_temp/Ma/software/src/general_funcitons/file_operation/results.txt"

#rm ./gpu_mcs_eclipse
#rm /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/scatterMCS/*
#read -p "Press enter to continue"
#./gpu_mcs_eclipse >$LOG_PATH
#wait 
#vi $LOG_PATH 
##vi /data/PET/mr_pet_temp/Ma/software/data/gpupet/phantom/cubic_phantom_ma/scatterMCS/cubic_phantom_gpuSimu_0_log


