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

#include <iostream>
#include <string>
#include <fstream>

#include "util/settings.h"
#include "util/NumType.h"

namespace dso
{
	void settingsDefault(int preset);

	/// <summary>
	/// 解析命令行输入参数
	/// </summary>
	/// <param name="arg">输入参数</param>
	void parseArgument(char* arg);

	/// <summary>
	/// 将四元数转换为旋转矩阵
	/// </summary>
	/// <param name="q">四元数</param>
	/// <returns>旋转矩阵</returns>
	Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q);

	/// <summary>
	/// 获取KITTI数据集的真实轨迹信息
	/// </summary>
	void getGroundtruth_kitti();

	/// <summary>
	/// 获取EUROC数据集的真实轨迹信息
	/// </summary>
	void getGroundtruth_euroc();

	/// <summary>
	/// 获取EUROC数据集IMU信息
	/// </summary>
	void getIMUdata_euroc();

	/// <summary>
	/// 获取双目相机外参信息
	/// </summary>
	void getTstereo();

	/// <summary>
	/// 获取IMU内参信息
	/// </summary>
	void getIMUinfo();

	/// <summary>
	/// 获取数据集时间戳信息
	/// </summary>
	void getPicTimestamp();
}