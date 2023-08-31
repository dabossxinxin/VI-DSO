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

#include "vector"
#include <iostream>
#include <fstream>

#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include "FullSystem/HessianBlocks.h"

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
	enum ResLocation
	{
		ACTIVE = 0,
		LINEARIZED,
		MARGINALIZED,
		NONE
	};

	enum ResState
	{
		INNER = 0,		// 残差的值小于阈值视为内点
		OOB,			// 点不在主帧或目标帧视野中
		OUTLIER			// 残差的值大于阈值视为外点
	};

	struct FullJacRowT
	{
		Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
	};

	// 地图点与关键帧之间的光度残差
	class PointFrameResidual
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		EFResidual* efResidual;				// 与EFResidual互相持有对方指针实现通信			
		bool stereoResidualFlag = false;	// 当前残差是否为双目残差的标志位
		static int instanceCounter;			// 残差数量计数的静态成员变量

		ResState state_state;				// TODO:当前残差的状态
		double state_energy;
		ResState state_NewState;
		double state_NewEnergy;
		double state_NewEnergyWithOutlier;

		PointHessian* point;				// 构造当前残差的地图点
		FrameHessian* host;					// 构建当前残差地图点的主帧
		FrameHessian* target;				// 构建当前残差地图点的投影帧
		RawResidualJacobian* J;				// 当前残差关于所求优化变量的雅可比

		void setState(ResState s) { state_state = s; }

		bool isNew;

		Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
		Vec3f centerProjectedTo;

		~PointFrameResidual();
		PointFrameResidual();
		PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);

		double linearize(CalibHessian* HCalib);
		double linearizeStereo(CalibHessian* HCalib);

		void resetOOB()
		{
			state_NewEnergy = state_energy = 0;
			state_NewState = ResState::OUTLIER;
			setState(ResState::INNER);
		};

		void applyRes(bool copyJacobians);
		void debugPlot();
	};
}