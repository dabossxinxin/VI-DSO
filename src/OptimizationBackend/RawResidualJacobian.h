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

#include "util/NumType.h"

namespace dso
{
	/// <summary>
	/// 系统中特征光度残差对相关变量的导数
	/// 特征包含中心点以及周围相邻7个点一起组成的Pattern
	/// </summary>
	struct RawResidualJacobian
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		EIGEN_ALIGN16 VecNRf resF;			// 特征在target帧中与host帧中的光度残差
		EIGEN_ALIGN16 Vec6f Jpdxi[2];		// 特征在target帧中的像素坐标相对于相机位姿的雅可比
		EIGEN_ALIGN16 VecCf Jpdc[2];		// 特征在target帧中的像素坐标相对于相机内参的雅可比
		EIGEN_ALIGN16 Vec2f Jpdd;			// 特征在target帧中的像素坐标相对于在host帧中逆深度的雅可比
		EIGEN_ALIGN16 VecNRf JIdx[2];		// 特征在target帧中的光度相对于在target帧中像素坐标的雅可比
		EIGEN_ALIGN16 VecNRf JabF[2];		// 特征光度残差相对于target帧与host帧之间的光度变换的雅可比

		EIGEN_ALIGN16 Mat22f JIdx2;			// JIdx^T * JIdx
		EIGEN_ALIGN16 Mat22f JabJIdx;		// Jab^T * JIdx
		EIGEN_ALIGN16 Mat22f Jab2;			// Jab^T * Jab
	};
}