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
#include <locale.h>
#include <stdlib.h>
#include <stdio.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/NumType.h"
#include "util/systemInput.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/MatrixAccumulators.h"

/// <summary>
/// ���������в���
/// </summary>
/// <param name="argc">��������</param>
/// <param name="argv">��������</param>
void loadArgument(int argc, char** argv)
{
	for (int it = 1; it < argc; ++it)
		dso::parseArgument(argv[it]);
}

int main(int argc, char** argv)
{
	loadArgument(argc, argv);

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

	ImageFolderReader* readerLeft = NULL, * readerRight = NULL;
	readerLeft = new ImageFolderReader(input_sourceLeft, input_calibLeft, input_gammaCalib, input_vignette);
	if (setting_useStereo)
		readerRight = new ImageFolderReader(input_sourceRight, input_calibRight, input_gammaCalib, input_vignette);
	else
		readerRight = new ImageFolderReader(input_sourceLeft, input_calibLeft, input_gammaCalib, input_vignette);

	readerLeft->setGlobalCalibration();
	readerRight->setGlobalCalibration();

	if (setting_photometricCalibration > 0 && readerLeft->getPhotometricGamma() == NULL)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}

	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(readerLeft->getPhotometricGamma());

	IOWrap::PangolinDSOViewer* viewer = NULL;
	if (!setting_disableAllDisplay)
	{
		viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
		fullSystem->outputWrapper.emplace_back(viewer);
	}

	// MacOS����ʾ�̱߳��������߳�������
	// ��ζ�ż����̱߳��������ڷ�֧�߳���
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

			timedso tv_start;
			gettimeofday(&tv_start, NULL);
			double sInitializerOffset = 0;

			for (int idx = 0; idx < (int)idsToPlayLeft.size(); ++idx)
			{
				if (!fullSystem->initialized)
				{
					gettimeofday(&tv_start, NULL);
					sInitializerOffset = timesToPlayAtLeft[idx];
				}

				// �����������������ж�Ӧ��ͼ��
				int idRight = -1;
				int idLeft = idsToPlayLeft[idx];
				double timestampRight = 0;
				double timestampLeft = pic_time_stamp[idLeft];

				if (setting_useStereo)
				{
					if (pic_time_stamp_r.size() > 0)
					{
						for (int id = 0; id < pic_time_stamp_r.size(); ++id)
						{
							timestampRight = pic_time_stamp_r[id];
							if (timestampRight >= timestampLeft ||
								std::fabs(timestampRight - timestampLeft) < 0.01)
							{
								idRight = id;
								break;
							}
						}
					}
					if (std::fabs(timestampRight - timestampLeft) > 0.01) continue;
				}

				ImageAndExposure* imgLeft = NULL, * imgRight = NULL;
				imgLeft = readerLeft->getImage(idLeft);
				if (!setting_useStereo) imgRight = imgLeft;
				else imgRight = readerRight->getImage(idRight);

				fullSystem->addActiveFrame(imgLeft, imgRight, idLeft);

				delete imgLeft;
				if (imgLeft != imgRight) delete imgRight;
				imgLeft = imgRight = NULL;

				// 1��ϵͳ��ʼ��ʧ�ܲ���ϵͳ�ոչ������ã������³�ʼ��
				// 2���ڽ������ֶ����������õ�ǰϵͳ�������³�ʼ��
				if (fullSystem->initFailed || setting_fullResetRequested)
				{
					if (idx < 250 || setting_fullResetRequested)
					{
						printf("- DSO RESETTING!\n");

						auto& wraps = fullSystem->outputWrapper;
						delete fullSystem;

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
					printf("- DSO LOST!\n");
					break;
				}
			}

			fullSystem->blockUntilMappingIsFinished();
			timedso tv_end;
			gettimeofday(&tv_end, NULL);
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

	if (viewer != NULL)
		viewer->run();
	runthread.join();

	for (auto ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("- DELETE FULLSYSTEM!\n");
	delete fullSystem;
	fullSystem = NULL;

	printf("- DELETE READER!\n");
	delete readerLeft;
	if (readerLeft != readerRight) delete readerRight;
	readerLeft = readerRight = NULL;

	printf("- EXIT NOW!\n");
	return 0;
}