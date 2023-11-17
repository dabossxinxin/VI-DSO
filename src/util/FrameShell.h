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
#include "algorithm"

namespace dso
{
	class FrameShell
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		int id; 			// 帧全局ID，从0开始递增
		int incomingId;		// 将图像输入算法时设置的图像ID
		int imuStampId;		// 图像帧时间戳在IMU数据中的索引
		double timestamp;	// 将图像输入算法时设置的图像时间戳

		// set once after tracking
		SE3 camToTrackingRef;
		FrameShell* trackingRef;

		// constantly adapted.
		SE3 camToWorld;				// Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
		AffLight aff_g2l;
		bool poseValid;

		int statistics_outlierResOnThis;	// 记录当前帧中管理的无效残差的数量		
		int statistics_goodResOnThis;		// 记录当前帧中管理的有效残差的数量
		int marginalizedAt;					// 记录当前帧被边缘化时系统最后一帧关键帧的全局序号
		double movedByOpt;					// TODO

		Vec3 velocity = Vec3::Zero();
		Vec3 bias_g = Vec3::Zero();
		Vec3 bias_a = Vec3::Zero();
		Vec3 delta_bias_g = Vec3::Zero();
		Vec3 delta_bias_a = Vec3::Zero();

		inline FrameShell()
		{
			id = 0;
			imuStampId = -1;
			incomingId = -1;
			poseValid = true;
			camToWorld = SE3();
			timestamp = 0;
			marginalizedAt = -1;
			movedByOpt = 0;
			statistics_outlierResOnThis = statistics_goodResOnThis = 0;
			trackingRef = 0;
			camToTrackingRef = SE3();
		}
	};
}