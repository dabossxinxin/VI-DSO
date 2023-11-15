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
	/// �����������������
	/// </summary>
	/// <param name="arg">�������</param>
	void parseArgument(char* arg);

	/// <summary>
	/// ����Ԫ��ת��Ϊ��ת����
	/// </summary>
	/// <param name="q">��Ԫ��</param>
	/// <returns>��ת����</returns>
	Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q);

	/// <summary>
	/// ��ȡKITTI���ݼ�����ʵ�켣��Ϣ
	/// </summary>
	void getGroundtruth_kitti();

	/// <summary>
	/// ��ȡEUROC���ݼ�����ʵ�켣��Ϣ
	/// </summary>
	void getGroundtruth_euroc();

	/// <summary>
	/// ��ȡEUROC���ݼ�IMU��Ϣ
	/// </summary>
	void getIMUdata_euroc();

	/// <summary>
	/// ��ȡ˫Ŀ��������Ϣ
	/// </summary>
	void getTstereo();

	/// <summary>
	/// ��ȡIMU�ڲ���Ϣ
	/// </summary>
	void getIMUinfo();

	/// <summary>
	/// ��ȡ���ݼ�ʱ�����Ϣ
	/// </summary>
	void getPicTimestamp();
}