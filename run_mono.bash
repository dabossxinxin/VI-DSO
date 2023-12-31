#!/bin/bash
#usage: python xxx.py file_name
dataname="MH_01_easy"
#dataname="MH_02_easy"
#dataname="MH_03_medium"
#dataname="MH_04_difficult"
#dataname="MH_05_difficult"
#dataname="V2_03_difficult"
#dataname="V2_01_easy"
#dataname="V2_02_medium"
#dataname="V1_02_medium"
#dataname="V1_01_easy"
#dataname="V1_03_difficult"
    # run dso
     ./build/bin/Debug/dso_dataset \
 	  files0=../../Data/${dataname}/mav0/cam0/data.zip \
 	  calib0=./calib/euroc/cam0.txt \
	  imu_info=./calib/euroc/IMU_info.txt \
	  groundtruth=../../Data/${dataname}/mav0/state_groundtruth_estimate0/data.csv \
	  imudata=../../Data/${dataname}/mav0/imu0/data.csv \
	  pic_timestamp0=../../Data/${dataname}/mav0/cam0/data.csv \
 	  preset=0 mode=1 \
	  quiet=1 nomt=1 \
	  use_stereo=0\
	  imu_weight=6 imu_weight_tracker=0.6 stereo_weight=0

