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
#include <thread>
#include <stdlib.h>
#include <stdio.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include "util/settings.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/NumType.h"
#include "util/systemInput.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"

/// <summary>
/// 加载命令行参数
/// </summary>
/// <param name="argc">参数数量</param>
/// <param name="argv">参数内容</param>
void loadArgument(int argc, char** argv)
{
	for (int it = 1; it < argc; ++it)
		dso::parseArgument(argv[it]);
}

/// <summary>
/// 设置系统运行参数
/// </summary>
/// <param name="folder">数据集文件夹路径</param>
void setArgument(const std::string folder)
{
	input_sourceLeft = folder + "/MH_01_easy/mav0/cam0/data.zip";
	input_calibLeft = folder + "/MH_01_easy/mav0/cam0/cam0.txt";
	input_calibImu = folder + "/MH_01_easy/mav0/imu0/imu0.txt";
	input_gtPath = folder + "/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
	input_imuPath = folder + "/MH_01_easy/mav0/imu0/data.csv";
	input_picTimestampLeft = folder + "/MH_01_easy/mav0/cam0/data.csv";

	setting_debugout_runquiet = true;
	setting_photometricCalibration = 0;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;
	setting_useStereo = false;
	setting_imuWeightNoise = 6;
	setting_imuWeightTracker = 0.6;
	setting_stereoWeight = 0;

	dso::settingsDefault(0);
}

int main(int argc, char** argv)
{
	//loadArgument(argc, argv);
#if defined(_WIN_)
    setArgument("E:/TumData");
#elif defined(_OSX_)
    setArgument("/Users/liuxianxian/Desktop");
#endif

	dso::getTstereo();
	dso::getIMUinfo();
	dso::getIMUdata_euroc();
	dso::getGroundtruth_euroc();
	dso::getPicTimestamp();

	setting_imuWeightNoise = 3;
	setting_imuWeightTracker = 0.1;
	setting_stereoWeight = 2;

	setting_gravityNorm = 9.81;
	setting_useImu = false;
	setting_imuTrackFlag = true;
	setting_imuTrackReady = false;

	setting_useOptimize = true;
	setting_useDynamicMargin = true;
	setting_initialIMUHessian = 0;
	setting_initialScaleHessian = 0;
	setting_initialbaHessian = 0;
	setting_initialbgHessian = 0;

	setting_dynamicMin = sqrt(1.1);
	setting_margWeightFacImu = 0.25;

	ImageFolderReader* readerLeft = nullptr, * readerRight = nullptr;
	readerLeft = new ImageFolderReader(input_sourceLeft, input_calibLeft, input_gammaCalib, input_vignette);
	if (setting_useStereo)
		readerRight = new ImageFolderReader(input_sourceRight, input_calibRight, input_gammaCalib, input_vignette);
	else
		readerRight = new ImageFolderReader(input_sourceLeft, input_calibLeft, input_gammaCalib, input_vignette);

	readerLeft->setGlobalCalibration();
	readerRight->setGlobalCalibration();

	if (setting_photometricCalibration > 0 && readerLeft->getPhotometricGamma() == nullptr)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}

	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(readerLeft->getPhotometricGamma());

	IOWrap::PangolinDSOViewer* viewer = nullptr;
	if (!setting_disableAllDisplay)
	{
		viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
		fullSystem->outputWrapper.emplace_back(viewer);
	}

	// MacOS中显示线程必须在主线程中运行
	// 意味着计算线程必须声明在分支线程中
	std::thread runthread([&]()
		{
			std::vector<int> idsToPlayLeft, idsToPlayRight;
			std::vector<double> timesToPlayAtLeft, timesToPlayAtRight;

			for (int idx = 0; idx < readerLeft->getNumImages(); ++idx)
			{
				idsToPlayLeft.emplace_back(idx);
				if (timesToPlayAtLeft.empty())
					timesToPlayAtLeft.emplace_back((double)0);
				else
					timesToPlayAtLeft.emplace_back(timesToPlayAtLeft.back());
			}

			for (int idx = 0; idx < readerRight->getNumImages(); ++idx)
			{
				idsToPlayRight.emplace_back(idx);
				if (timesToPlayAtRight.empty())
					timesToPlayAtRight.emplace_back((double)0);
				else
					timesToPlayAtRight.emplace_back(timesToPlayAtRight.back());
			}

#if defined(_WIN_)
			timedso tv_start;
#elif defined(_OSX_)
			timeval tv_start;
#endif
			gettimeofday(&tv_start, nullptr);
			double sInitializerOffset = 0;

			for (int idx = 0; idx < (int)idsToPlayLeft.size(); ++idx)
			{
				if (!fullSystem->initialized)
				{
					gettimeofday(&tv_start, nullptr);
					sInitializerOffset = timesToPlayAtLeft[idx];
				}

				// 搜索左相机在右相机中对应的图像
				int idRight = -1;
				int idLeft = idsToPlayLeft[idx];
				double timestampRight = 0;
				double timestampLeft = input_picTimestampLeftList[idLeft];

				if (setting_useStereo)
				{
					if (!input_picTimestampRightList.empty())
					{
						for (int id = 0; id < input_picTimestampRightList.size(); ++id)
						{
							timestampRight = input_picTimestampRightList[id];
							if (timestampRight >= timestampLeft ||
								std::abs(timestampRight - timestampLeft) < 0.01)
							{
								idRight = id;
								break;
							}
						}
					}
					if (std::abs(timestampRight - timestampLeft) > 0.01) continue;
				}

				ImageAndExposure* imgLeft = nullptr, * imgRight = nullptr;
				imgLeft = readerLeft->getImage(idLeft);
				if (!setting_useStereo) imgRight = imgLeft;
				else imgRight = readerRight->getImage(idRight);

				fullSystem->addActiveFrame(imgLeft, imgRight, idLeft);

				delete imgLeft;
				if (imgLeft != imgRight) delete imgRight;
				imgLeft = imgRight = nullptr;

				// 1、系统初始化失败并且系统刚刚工作不久，需重新初始化
				// 2、在界面中手动设置了重置当前系统，需重新初始化
				if (fullSystem->initFailed || setting_fullResetRequested)
				{
					if (idx < 250 || setting_fullResetRequested)
					{
						printf("- DSO RESETTING!\n");

						auto wraps = fullSystem->outputWrapper;
						SAFE_DELETE(fullSystem);

						for (IOWrap::Output3DWrapper* ow : wraps) ow->reset();

						fullSystem = new FullSystem();
						fullSystem->setGammaFunction(readerLeft->getPhotometricGamma());

						fullSystem->outputWrapper = wraps;

						setting_fullResetRequested = false;
						first_track_flag = false;
					}
				}

				if (fullSystem->isLost)
				{
					printf("INFO: DSO LOST!\n");
					break;
				}
			}

			fullSystem->blockUntilMappingIsFinished();
#if defined(_WIN_)
			timedso tv_end;
#elif defined(_OSX_)
			timeval tv_end;
#endif
			gettimeofday(&tv_end, nullptr);
			fullSystem->printResult("result.txt");

			int numFramesProcessed = std::abs(idsToPlayLeft[0] - idsToPlayLeft.back());
			double numSecondsProcessed = fabs(readerLeft->getTimestamp(idsToPlayLeft[0]) - readerLeft->getTimestamp(idsToPlayLeft.back()));
			double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
			printf("\n======================"
				"\n%d Frames (%.1f fps)"
				"\n%.2fms per frame (multi core); "
				"\n%.3fx (multi core); "
				"\n======================\n\n",
				numFramesProcessed, numFramesProcessed / numSecondsProcessed,
				MilliSecondsTakenMT / (float)numFramesProcessed,
				1000 / (MilliSecondsTakenMT / numSecondsProcessed));
		});

	if (viewer != nullptr)
		viewer->run();
	runthread.join();

	for (auto ow : fullSystem->outputWrapper)
	{
		ow->join();
		SAFE_DELETE(ow);
	}

	printf("INFO: DELETE FULLSYSTEM!\n");
	SAFE_DELETE(fullSystem);

	printf("INFO: DELETE READER!\n");
	delete readerLeft;
	if (readerLeft != readerRight) delete readerRight;
	readerLeft = readerRight = nullptr;

	printf("- EXIT NOW!\n");
	return 0;
}