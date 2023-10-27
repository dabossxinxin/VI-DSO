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

#include <vector>
#include <math.h>

#include "util/NumType.h"

#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
	class EFFrame;
	class EFPoint;
	class EFResidual;
		
	class FrameHessian;
	class PointHessian;
	class CalibHessian;

	class PointFrameResidual;
	class EnergyFunctional;

	/// <summary>
	/// 后端中管理的观测残差数据结构
	/// </summary>
	class EFResidual
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
			data(org), point(point_), host(host_), target(target_)
		{
			isLinearized = false;
			isActiveAndIsGoodNEW = false;
			J = new RawResidualJacobian();
			assert(((long)this) % 16 == 0);
			assert(((long)J) % 16 == 0);
		}
		inline ~EFResidual()
		{
			delete J;
		}

		void takeDataF();

		void fixLinearizationF(EnergyFunctional* ef);

		// structural pointers
		PointFrameResidual* data;		// 当前残差对应的观测特征在前端跟踪中的表示
		int hostIDX;					// 当前残差对应观测特征的主帧在滑窗关键帧序列中的ID
		int targetIDX;					// 当前残差对应观测特征的目标帧在滑窗关键帧序列中的ID
		EFPoint* point;					// 当前残差对应的观测特征在滑窗优化中的表示
		EFFrame* host;					// 当前残差对应观测特征主帧在滑窗优化中的表示
		EFFrame* target;				// 当前残差对应观测特征目标帧在滑窗优化中的表示
		int idxInAll;					// 观测残差在特征残差序列中的序号

		RawResidualJacobian* J;			// 当前观测残差对内参、位姿、光度参数以及逆深度的雅可比

		// 	VecNRf res_toZeroF;
		// 	Vec8f JpJdF=Vec8f::Zero();
		EIGEN_ALIGN16 VecNRf res_toZeroF;
		EIGEN_ALIGN16 Vec8f JpJdF = Vec8f::Zero();		// p表示帧位姿以及光度参数、d表示特征逆深度

		// status.
		bool isLinearized;

		// if residual is not OOB & not OUTLIER & should be used during accumulations
		bool isActiveAndIsGoodNEW;
		inline const bool &isActive() const { return isActiveAndIsGoodNEW; }
	};

	enum EFPointStatus { PS_GOOD = 0, PS_MARGINALIZE, PS_DROP };

	class EFPoint
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFPoint(PointHessian* d, EFFrame* host_) : data(d), host(host_)
		{
			takeData();
			stateFlag = EFPointStatus::PS_GOOD;
		}
		void takeData();

		PointHessian* data;				// 特征在前端中的表示形式

		float priorF;					// 特征先验信息
		float deltaF;					// 特征在滑窗优化中的优化增量
			
		int idxInPoints;				// 特征在对应主帧管理的特征序列中的序号
		EFFrame* host;					// 特征对应的主帧

		std::vector<EFResidual*> residualsAll;	// 特征管理的观测残差序列

		float bdSumF;
		float HdiF = 1e-3;		// TODO
		float Hdd_accLF;		// 特征线性化状态下的逆深度Hessian
		VecCf Hcd_accLF;		// 特征线性化状态下的逆深度&相机内参Hessian
		float bd_accLF;			// 特征线性化状态下的逆深度b
		float Hdd_accAF;		// 特征激活状态下的逆深度Hessian
		VecCf Hcd_accAF;		// 特征激活状态下的逆深度&相机内参Hessian
		float bd_accAF;			// 特征激活状态下的逆深度b

		EFPointStatus stateFlag;
	};

	/// <summary>
	/// 后端中管理的关键帧数据形式
	/// </summary>
	class EFFrame
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFFrame(FrameHessian* d) : data(d)
		{
			takeData();
		}
		void takeData();

		Vec8 prior;				// prior hessian (diagonal)
		Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
		Vec8 delta;				// state - state_zero.

		std::vector<EFPoint*> points;	// 关键帧管理的特征序列
		FrameHessian* data;				// 关键帧在前端中的表示

		int idx;						// 关键帧在滑窗关键帧中的序号
		int frameID;					// 关键帧在全局中的帧序号
		bool m_flag = false;
	};
}