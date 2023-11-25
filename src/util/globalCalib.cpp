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
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "util/globalCalib.h"

namespace dso
{
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

	float wM3G;
	float hM3G;

	void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K)
	{
		int wlvl = w;
		int hlvl = h;
		pyrLevelsUsed = 1;
		while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /= 2;
			hlvl /= 2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
			pyrLevelsUsed - 1, wlvl, hlvl);
		if (wlvl > 100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
				"using not enough pyramid levels.\n"
				"Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if (pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
				"I need higher resolution.\n"
				"I will probably segfault.\n");
		}

		wM3G = w - 3;
		hM3G = h - 3;

		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0, 0);
		fyG[0] = K(1, 1);
		cxG[0] = K(0, 2);
		cyG[0] = K(1, 2);
		KiG[0] = KG[0].inverse();
		fxiG[0] = KiG[0](0, 0);
		fyiG[0] = KiG[0](1, 1);
		cxiG[0] = KiG[0](0, 2);
		cyiG[0] = KiG[0](1, 2);

		// 设置图像在各层金字塔的图像宽高以及模拟相机内参
		for (int level = 1; level < pyrLevelsUsed; ++level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[level] = fxG[level - 1] * 0.5;
			fyG[level] = fyG[level - 1] * 0.5;
			cxG[level] = (cxG[0] + 0.5) / ((int)1 << level) - 0.5;
			cyG[level] = (cyG[0] + 0.5) / ((int)1 << level) - 0.5;

			KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;	// synthetic
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0, 0);
			fyiG[level] = KiG[level](1, 1);
			cxiG[level] = KiG[level](0, 2);
			cyiG[level] = KiG[level](1, 2);
		}
	}

	void preintergrate(IMUPreintegrator& handle, const Vec3& bg, const Vec3& ba, 
		const double timeStart, const double timeEnd)
	{
		double delta_t = 0;
		int imuStartStamp = -1;
		imuStartStamp = findNearestIdx(input_imuTimestampList, timeStart);
		assert(imuStartStamp != -1);

		// 从当前帧时刻到下一帧时刻内的IMU测量值预积分
		while (true)
		{
			if (input_imuTimestampList[imuStartStamp + 1] < timeEnd)
				delta_t = input_imuTimestampList[imuStartStamp + 1] - input_imuTimestampList[imuStartStamp];
			else
			{
				delta_t = timeEnd - input_imuTimestampList[imuStartStamp];
				if (delta_t < 1e-6) break;
			}

			handle.update(input_gryList[imuStartStamp] - bg, input_accList[imuStartStamp] - ba, delta_t);
			if (input_imuTimestampList[imuStartStamp + 1] >= timeEnd) break;
			imuStartStamp++;
		}
	}

    void saveDataTxt(const VecX& data, const std::string path, const int row)
    {
        std::ofstream fout;
        fout.open(path.c_str());

        fout << std::endl << std::endl;
        for (int it = 0; it < row; ++it)
            fout << data[it] << std::endl;
        fout << std::endl << std::endl;

        fout.close();
    }

    void saveDataTxt(const MatXX& data, const std::string path, const int row, const int col)
    {
        std::ofstream fout;
        fout.open(path.c_str());

        fout << std::endl << std::endl;
        for (int r = 0; r < row; ++r)
        {
			for (int c = 0; c < col; ++c)
				fout << data(r, c) << "\t";
            fout << std::endl;
        }
        fout << std::endl << std::endl;

        fout.close();
    }

//    bool haveNanData(const Mat88& data)
//    {
//        for (int r = 0; r < 8; ++r)
//        {
//            for (int c = 0; c < 8; ++c)
//            {
//                if (std::isnan(data(r,c)))
//                    return true;
//            }
//        }
//
//        return false;
//    }

//    bool haveNanData(const Mat1717& data)
//    {
//        for (int r = 0; r < 17; ++r)
//        {
//            for (int c = 0; c < 17; ++c)
//            {
//                if (std::isnan(data(r,c)))
//                    return true;
//            }
//        }
//
//        return false;
//    }

//    bool haveNanData(const VecX& data, const int row)
//    {
//        for (int it = 0; it < row; ++it)
//        {
//            if (std::isnan(data[it]))
//                return true;
//        }
//
//        return false;
//    }

//    bool haveNanData(const MatXX& data, const int row, const int col)
//    {
//        for (int r = 0; r < row; ++r)
//        {
//            for (int c = 0; c < col; ++c)
//            {
//                if (std::isnan(data(r,c)))
//                    return true;
//            }
//        }
//
//        return false;
//    }
}