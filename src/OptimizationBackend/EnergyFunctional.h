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

#include <map>
#include <vector>
#include <math.h>

#include "util/NumType.h"
#include "util/IndexThreadReduce.h"

#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/IMUPreintegrator.h"

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

namespace dso
{
	extern bool EFAdjointsValid;		// 优化函数中参数相对量对绝对量的雅可比是否求解完成
	extern bool EFIndicesValid;			// 优化函数中滑窗关键帧以及管理特征的序号是否编好
	extern bool EFDeltaValid;			// 优化函数中滑窗关键帧以及管理特征的增量是否设置好

	class EnergyFunctional
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		friend class EFFrame;
		friend class EFPoint;
		friend class EFResidual;
		friend class AccumulatedTopHessian;
		friend class AccumulatedTopHessianSSE;
		friend class AccumulatedSCHessian;
		friend class AccumulatedSCHessianSSE;

		EnergyFunctional();
		~EnergyFunctional();

		EFResidual* insertResidual(PointFrameResidual* r);
		EFFrame* insertFrame(FrameHessian* fh, CalibHessian* Hcalib);
		EFPoint* insertPoint(PointHessian* ph);

		void dropResidual(EFResidual* r);
		void marginalizeFrame(EFFrame* fh);
		void marginalizeFrame_imu(EFFrame* fh);
		void removePoint(EFPoint* ph);

		void marginalizePointsF();
		void dropPointsF();

		// 滑窗优化函数
		void solveSystemF(int iteration, double lambda, CalibHessian* HCalib);
		double calcMEnergyF();
		double calcLEnergyF_MT();

		void makeIDX();

		void setDeltaF(CalibHessian* HCalib);

		void setAdjointsF(CalibHessian* Hcalib);

		std::vector<EFFrame*> frames;		// 滑窗中所有的关键帧
		int nPoints;						// 参与滑窗优化的特征数量
		int nFrames;						// 参与滑窗优化的关键帧数量
		int nResiduals;						// 参与滑窗优化的光度残差数量

		MatXX HM_visual;					// 视觉部分边缘化的H
		VecX bM_visual;						// 视觉部分边缘化的b
		    
		MatXX HM_imu;						// 惯导部分边缘化的H
		VecX bM_imu;						// 惯导部分边缘化的b

		MatXX HM_bias;
		VecX bM_bias;

		MatXX HM_imu_half;
		VecX bM_imu_half;

		double s_middle = 1;
		double s_last = 1;
		double d_now = sqrt(1.1);
		double d_half = sqrt(1.1);
		bool side_last = true;		//for dynamic margin: true: upper s_middle false: below s_middle

		int resInA;
		int resInL;
		int resInM;
		MatXX lastHS;				// 上一步优化迭代视觉Hessian
		VecX lastbS;				// 上一步优化迭代视觉b
		VecX lastX;					// 上一步优化迭代滑窗关键帧内参、位姿以及光度增量

		std::vector<VecX> lastNullspaces_forLogging;		// 上一优化迭代步骤的零空间
		std::vector<VecX> lastNullspaces_pose;				// 上一优化迭代步骤的位姿零空间
		std::vector<VecX> lastNullspaces_scale;				// 上一优化迭代步骤的尺度零空间
		std::vector<VecX> lastNullspaces_affA;				// 上一优化迭代步骤的光度参数a零空间
		std::vector<VecX> lastNullspaces_affB;				// 上一优化迭代步骤的光度参数b零空间

		IndexThreadReduce<Vec10>* red;						// 滑窗优化类内部使用的多线程工具handle

		std::map<uint64_t,
			Eigen::Vector2i,
			std::less<uint64_t>,
			Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
		> connectivityMap;									// TODO：维护滑窗关键帧之间的共视关系
	private:

		VecX getStitchedDeltaF() const;

		// 根据求解得到的相机位姿以及相机内参更新量计算特征点逆深度的更新量 
		void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
		void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

		void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

		void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

        void schurHessianImu(EFFrame* fh, MatXX& H, VecX& b);
        void schurHessianVisual(EFFrame* fh, MatXX& H, VecX& b);

		void calcIMUHessian(MatXX &H, VecX &b);

		void orthogonalize(VecX* b, MatXX* H);
		Mat18f* adHTdeltaF;		// hostToTarget位姿、光度变换参数 [0~5：相机位姿，6、7：光度参数]

		Mat88* adHost;			// hostToTarget位姿、光度变换对host位姿、光度的雅可比 [double]
		Mat88* adTarget;		// hostToTarget位姿、光度变换对target位姿、光度的雅可比 [double]
		Mat88f* adHostF;		// hostToTarget位姿、光度变换对host位姿、光度的雅可比 [float]
		Mat88f* adTargetF;		// hostToTarget位姿、光度变换对target位姿、光度的雅可比 [float]

		VecCf cDeltaF;			// 相机内参参数增量
		VecC cPrior;			// 相机内参参数先验值 [double]
		VecCf cPriorF;			// 相机内参参数先验值 [float]
		
		AccumulatedTopHessianSSE* accSSE_top_L;		// 计算滑窗中激活特征的Heesian信息
		AccumulatedTopHessianSSE* accSSE_top_A;		// 计算滑窗中特征的除逆深度外的Hessian信息
		AccumulatedSCHessianSSE* accSSE_bot;		// 计算滑窗中边缘化特征的舒尔补信息

		std::vector<EFPoint*> allPoints;			// 参与优化计算的所有特征
		std::vector<EFPoint*> allPointsToMarg;		// 参与优化计算的所有边缘化特征

		float currentLambda;						// 当前滑窗优化迭代步骤的阻尼因子
	};
}