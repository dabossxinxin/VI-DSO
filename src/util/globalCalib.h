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
	/// DSO系统设置全局相机内参数
	/// </summary>
	/// <param name="w">相机图像宽度</param>
	/// <param name="h">相机图像高度</param>
	/// <param name="K">相机内参</param>
	void setGlobalCalib(int w, int h, const Eigen::Matrix3f& K);
	
	/// <summary>
	/// DSO系统两图像帧之间惯导数据预积分
	/// </summary>
	/// <param name="handle">预积分handle</param>
	/// <param name="bg">惯导数据陀螺仪偏置</param>
	/// <param name="ba">惯导数据加速度偏置</param>
	/// <param name="timeStart">惯导数据开始时间</param>
	/// <param name="timeEnd">惯导数据结束时间</param>
	void preintergrate(IMUPreintegrator& handle, const Vec3& bg, const Vec3& ba,
		const double timeStart, const double timeEnd);
}