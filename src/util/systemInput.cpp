/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/
#include "util/systemInput.h"

namespace dso
{
	void settingsDefault(int preset)
	{
		printf("\n=============== PRESET Settings: ===============\n");
		if (preset == 0 || preset == 1)
		{
			printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset == 0 ? "no " : "1x");

			setting_desiredImmatureDensity = 1500;
			setting_desiredPointDensity = 2000;
			setting_minFrames = 5;
			setting_maxFrames = 7;
			setting_maxOptIterations = 6;
			setting_minOptIterations = 1;

			setting_logStuff = false;
			setting_kfGlobalWeight = 1;						// original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set this para higher
			setting_maxShiftWeightT = 0.04f * (640 + 128);	// original is 0.04f * (640+480); this para is depend on the crop size.
			setting_maxShiftWeightR = 0.04f * (640 + 128);	// original is 0.00f * (640+480);
			setting_maxShiftWeightRT = 0.02f * (640 + 128);	// original is 0.02f * (640+480);
		}

		if (preset == 2 || preset == 3)
		{
			printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset == 0 ? "no " : "5x");

			setting_desiredImmatureDensity = 600;
			setting_desiredPointDensity = 800;
			setting_minFrames = 4;
			setting_maxFrames = 6;
			setting_maxOptIterations = 4;
			setting_minOptIterations = 1;

			benchmarkSetting_width = 424;
			benchmarkSetting_height = 320;

			setting_logStuff = false;
		}
		printf("==============================================\n");
	}

	void parseArgument(char* arg)
	{
		int option;
		float foption;
		char buf[1000];

		if (1 == sscanf(arg, "quiet=%d", &option))
		{
			if (option == 1)
			{
				setting_debugout_runquiet = true;
				printf("QUIET MODE, I'll shut up!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "preset=%d", &option))
		{
			settingsDefault(option);
			return;
		}

		if (1 == sscanf(arg, "nolog=%d", &option))
		{
			if (option == 1)
			{
				setting_logStuff = false;
				printf("DISABLE LOGGING!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "nogui=%d", &option))
		{
			if (option == 1)
			{
				setting_disableAllDisplay = true;
				printf("NO GUI!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "nomt=%d", &option))
		{
			if (option == 1)
			{
				setting_multiThreading = false;
				printf("NO MultiThreading!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "files0=%s", buf))
		{
			input_sourceLeft = buf;
			printf("loading data from %s!\n", input_sourceLeft.c_str());
			return;
		}

		if (1 == sscanf(arg, "files1=%s", buf))
		{
			input_sourceRight = buf;
			printf("loading data from %s!\n", input_sourceRight.c_str());
			return;
		}

		if (1 == sscanf(arg, "groundtruth=%s", buf))
		{
			input_gtPath = buf;
			printf("loading groundtruth from %s!\n", input_gtPath.c_str());
			return;
		}

		if (1 == sscanf(arg, "imudata=%s", buf))
		{
			input_imuPath = buf;
			printf("loading imudata from %s!\n", input_imuPath.c_str());
			return;
		}

		if (1 == sscanf(arg, "calib0=%s", buf))
		{
			input_calibLeft = buf;
			printf("loading camera calibration from %s!\n", input_calibLeft.c_str());
			return;
		}

		if (1 == sscanf(arg, "calib1=%s", buf))
		{
			input_calibRight = buf;
			printf("loading camera calibration from %s!\n", input_calibRight.c_str());
			return;
		}

		if (1 == sscanf(arg, "calib_stereo=%s", buf))
		{
			input_calibStereo = buf;
			printf("loading camera calibration from %s!\n", input_calibStereo.c_str());
			return;
		}

		if (1 == sscanf(arg, "calib_imu=%s", buf))
		{
			input_calibImu = buf;
			printf("loading imu calibration from %s!\n", input_calibImu.c_str());
			return;
		}

		if (1 == sscanf(arg, "pic_timestamp0=%s", buf))
		{
			input_picTimestampLeft = buf;
			printf("loading images timestamp from %s!\n", input_calibImu.c_str());
			return;
		}

		if (1 == sscanf(arg, "pic_timestamp1=%s", buf))
		{
			input_picTimestampRight = buf;
			printf("loading images timestamp from %s!\n", input_calibImu.c_str());
			return;
		}

		if (1 == sscanf(arg, "vignette=%s", buf))
		{
			input_vignette = buf;
			printf("loading vignette from %s!\n", input_vignette.c_str());
			return;
		}

		if (1 == sscanf(arg, "gamma=%s", buf))
		{
			input_gammaCalib = buf;
			printf("loading gammaCalib from %s!\n", input_gammaCalib.c_str());
			return;
		}

		if (1 == sscanf(arg, "imu_weight_noise=%f", &foption))
		{
			setting_imuWeightNoise = foption;
			return;
		}

		if (1 == sscanf(arg, "imu_weight_tracker=%f", &foption))
		{
			setting_imuWeightTracker = foption;
			return;
		}

		if (1 == sscanf(arg, "stereo_weight=%f", &foption))
		{
			setting_stereoWeight = foption;
			return;
		}

		if (1 == sscanf(arg, "use_stereo=%d", &option))
		{
			if (option == 0)
			{
				setting_useStereo = false;
				printf("Don't Use Stereo!\n");
			}
			else 
			{
				setting_useStereo = true;
				printf("Use Stereo!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "save_image=%d", &option))
		{
			if (option == 1)
			{
				setting_debugSaveImages = true;
				printf("SAVE IMAGES!\n");
			}
			return;
		}

		if (1 == sscanf(arg, "mode=%d", &option))
		{
			if (option == 0)
			{
				printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
			}
			if (option == 1)
			{
				printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
				setting_photometricCalibration = 0;
				setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
				setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			}
			if (option == 2)
			{
				printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
				setting_photometricCalibration = 0;
				setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
				setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
				setting_minGradHistAdd = 3;
			}
			return;
		}

		printf("could not parse argument \"%s\"!!!!\n", arg);
	}

	Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q)
	{
		Eigen::Matrix3d R = Eigen::Matrix3d::Zero();

		R(0, 0) = 1 - 2.0 * q(1) * q(1) - 2.0 * q(2) * q(2);
		R(0, 1) = 2.0 * (q(0) * q(1) - q(2) * q(3));
		R(0, 2) = 2.0 * (q(0) * q(2) + q(1) * q(3));

		R(1, 0) = 2.0 * (q(0) * q(1) + q(2) * q(3));
		R(1, 1) = -1 * q(0) * q(0) + q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
		R(1, 2) = 2.0 * (q(1) * q(2) - q(0) * q(3));

		R(2, 0) = 2.0 * (q(0) * q(2) - q(1) * q(3));
		R(2, 1) = 2.0 * (q(1) * q(2) + q(0) * q(3));
		R(2, 2) = -1 * q(0) * q(0) - q(1) * q(1) + q(2) * q(2) + q(3) * q(3);
		return R;
	}

	void getGroundtruth_kitti()
	{
		if (!input_gtPath.empty())
		{
			std::ifstream inf;
			inf.open(input_gtPath);
			std::string sline;
			std::getline(inf, sline);
			while (std::getline(inf, sline))
			{
				Mat33 R; Vec3 t;
				std::istringstream ss(sline);
				for (int i = 0; i < 3; ++i)
				{
					for (int j = 0; j < 3; ++j)
					{
						ss >> R(i, j);
					}
					ss >> t(i);
				}
				SE3 temp(R, t);
				input_gtPose.emplace_back(temp);
			}
			inf.close();
		}
		else
			printf("we get an empty path.\n");
	}

	void getGroundtruth_euroc()
	{
		if (!input_gtPath.empty())
		{
			std::ifstream inf;
			inf.open(input_gtPath);
			std::string sline;
			std::getline(inf, sline);
			while (std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				Vec4 q4;
				Vec3 t;
				Vec3 v;
				Vec3 bias_g;
				Vec3 bias_a;
				double time;
				ss >> time;
				time = time / 1e9;
				char temp;
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> t(i);
				}
				ss >> temp;
				ss >> q4(3);
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> q4(i);
				}
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> v(i);
				}
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> bias_g(i);
				}
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> bias_a(i);
				}
				Eigen::Matrix3d R_wb = quaternionToRotation(q4);
				SE3 pose0(R_wb, t);

				input_gtPose.emplace_back(pose0);
				input_gtVelocity.emplace_back(v);
				input_gtBiasG.emplace_back(bias_g);
				input_gtBiasA.emplace_back(bias_a);
				gt_time_stamp.emplace_back(time);
			}
			inf.close();
		}
		else
			printf("we get an empty path.\n");
	}

	void getIMUdata_euroc()
	{
		if (!input_imuPath.empty())
		{
			std::ifstream inf;
			inf.open(input_imuPath);
			std::string sline;
			std::getline(inf, sline);
			while (std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				Vec3 gyro, acc;
				double time;
				ss >> time;
				time = time / 1e9;
				char temp;
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> gyro(i);
				}
				for (int i = 0; i < 3; ++i)
				{
					ss >> temp;
					ss >> acc(i);
				}
				input_gryList.emplace_back(gyro);
				input_accList.emplace_back(acc);
				imu_time_stamp.emplace_back(time);
			}
			inf.close();
		}
		else
			printf("we get an empty path.\n");
	}

	void getTstereo()
	{
		if (!input_calibStereo.empty())
		{
			std::ifstream inf;
			inf.open(input_calibStereo);
			std::string sline;
			int line = 0;
			Mat33 R;
			Vec3 t;
			while (line < 3 && std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				for (int i = 0; i < 3; ++i)
				{
					ss >> R(line, i);
				}
				ss >> t(line);
				++line;
			}
			inf.close();
			SE3 temp(R, t);
			T_C0C1 = temp;
			T_C1C0 = temp.inverse();
		}
		else
			printf("we get an empty path.\n");
	}

	void getIMUinfo()
	{
		if (!input_calibImu.empty())
		{
			std::ifstream inf;
			inf.open(input_calibImu);
			std::string sline;
			int line = 0;
			Mat33 R;
			Vec3 t;
			Vec4 noise;
			while (line < 3 && std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				for (int i = 0; i < 3; ++i)
				{
					ss >> R(line, i);
				}
				ss >> t(line);
				++line;
			}
			std::getline(inf, sline);
			++line;
			while (line < 8 && std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				ss >> noise(line - 4);
				++line;
			}
			SE3 temp(R, t);
			T_BC = temp;

			GyrCov = Mat33::Identity() * noise(0) * noise(0) / 0.005;
			AccCov = Mat33::Identity() * noise(1) * noise(1) / 0.005;
			GyrRandomWalkNoise = Mat33::Identity() * noise(2) * noise(2);
			AccRandomWalkNoise = Mat33::Identity() * noise(3) * noise(3);

            std::cout << "INFO: T_BC: " << std::endl << T_BC.matrix() << std::endl;
            std::cout << "INFO: Noise: " << std::endl << noise.transpose() << std::endl;
			inf.close();
		}
		else
			printf("we get an empty path.\n");
	}

	void getPicTimestamp()
	{
		if (!input_picTimestampLeft.empty())
		{
			std::ifstream inf;
			inf.open(input_picTimestampLeft);
			std::string sline;
			std::getline(inf, sline);

			while (std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				double time;
				ss >> time;
				time = time / 1e9;
				pic_time_stamp.emplace_back(time);
			}
			inf.close();
		}
		else
			printf("we get an empty path.\n");

		if (!input_picTimestampRight.empty())
		{
			std::ifstream inf;
			inf.open(input_picTimestampRight);
			std::string sline;
			std::getline(inf, sline);
			while (std::getline(inf, sline))
			{
				std::istringstream ss(sline);
				double time;
				ss >> time;
				time = time / 1e9;
				pic_time_stamp_r.emplace_back(time);
			}
			inf.close();
		}
		else
			printf("we get an empty path.\n");
	}
}