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
#pragma once

#include "util/settings.h"
#include "util/NumType.h"
#include "FullSystem/IMUPreintegrator.h"

namespace dso
{
	extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
	extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	extern Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

	extern float wM3G;
	extern float hM3G;

	/// <summary>
	/// DSOϵͳ����ȫ������ڲ���
	/// </summary>
	/// <param name="w">���ͼ����</param>
	/// <param name="h">���ͼ��߶�</param>
	/// <param name="K">����ڲ�</param>
	void setGlobalCalib(int w, int h, const Eigen::Matrix3f& K);
	
	/// <summary>
	/// DSOϵͳ��ͼ��֮֡��ߵ�����Ԥ����
	/// </summary>
	/// <param name="handle">Ԥ����handle</param>
	/// <param name="bg">�ߵ�����������ƫ��</param>
	/// <param name="ba">�ߵ����ݼ��ٶ�ƫ��</param>
	/// <param name="timeStart">�ߵ����ݿ�ʼʱ��</param>
	/// <param name="timeEnd">�ߵ����ݽ���ʱ��</param>
	void preintergrate(IMUPreintegrator& handle, const Vec3& bg, const Vec3& ba,
		const double timeStart, const double timeEnd);

    /// <summary>
    /// ����������ݵ�txt�ļ�
    /// </summary>
    /// <param name="data">��������</param>
    /// <param name="path">����·��</param>
    /// <param name="row">����ά��</param>
	void saveDataTxt(const VecX& data, const std::string path, const int row);

    /// <summary>
    /// ����������ݵ�txt�ļ�
    /// </summary>
    /// <param name="data">��������</param>
    /// <param name="path">����·��</param>
    /// <param name="row">����ά��</param>
    /// <param name="col">����ά��</param>
    void saveDataTxt(const MatXX& data, const std::string path, const int row, const int col);

    //bool haveNanData(const Mat88& data);

    //bool haveNanData(const Mat1717& data);

    //bool haveNanData(const VecX& data, const int row);

    template <typename T>
    bool haveNanData(const T& data, const int row, const int col)
    {
        for (int r = 0; r < row; ++r)
        {
            for (int c = 0; c < col; ++c)
            {
                if (std::isnan(data(r,c)))
                    return true;
            }
        }

        return false;
    }
}