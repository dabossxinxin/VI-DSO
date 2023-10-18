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
#include <algorithm>

#include "util/settings.h"
#include "util/NumType.h"
#include "util/FrameShell.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/Residuals.h"
#include "FullSystem/IMUPreintegrator.h"
#include "FullSystem/HessianBlocks.h"

#include "OptimizationBackend/MatrixAccumulators.h"

namespace dso
{
	class CoarseTracker 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		CoarseTracker(int w, int h);
		~CoarseTracker();

		bool trackNewestCoarse(FrameHessian* newFrameHessian, SE3& lastToNew_out, AffLight& aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort, IOWrap::Output3DWrapper* wrap = 0);

		void setCTRefForFirstFrame(std::vector<FrameHessian*> frameHessians);
		void setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians, FrameHessian* fhRight, CalibHessian Hcalib);
		
		void makeCoarseDepthForFirstFrame(FrameHessian* fh);
		void makeK(CalibHessian* HCalib);

		bool debugPrint;		// 控制调试信息输出的标志位
		bool debugPlot;			// 控制调试信息绘制的标志位

		// 各金字塔层级内参信息
		Mat33f K[PYR_LEVELS];
		Mat33f Ki[PYR_LEVELS];
		float fx[PYR_LEVELS];
		float fy[PYR_LEVELS];
		float fxi[PYR_LEVELS];
		float fyi[PYR_LEVELS];
		float cx[PYR_LEVELS];
		float cy[PYR_LEVELS];
		float cxi[PYR_LEVELS];
		float cyi[PYR_LEVELS];

		// 各金字塔图像宽高
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
		void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

		FrameHessian* lastRef;			// 跟踪时的参考帧
		AffLight lastRef_aff_g2l;		// 跟踪时参考帧的光度变换参数
		FrameHessian* newFrame;			// 待跟踪的最新帧
		int refFrameID;					// 跟踪时参考帧的ID

		Vec5 lastResiduals;				// 各层金字塔残差信息
		Vec3 lastFlowIndicators;		// 第零层金字塔光流信息
		double firstCoarseRMSE;			// 首次跟踪第零层金字塔光度残差
		int pc_n[PYR_LEVELS];
	private:

		void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);
		
		Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		double calcIMUResAndGS(Mat66 &H_out, Vec6 &b_out, SE3 &refToNew, const IMUPreintegrator &IMU_preintegrator, Vec9 &res_PVPhi, double PointEnergy, double imu_track_weight);
		Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
		void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

		float* idepth[PYR_LEVELS];
		float* weightSums[PYR_LEVELS];
		float* weightSums_bak[PYR_LEVELS];

		// 跟踪时特征在参考帧上的信息
		float* pc_u[PYR_LEVELS];
		float* pc_v[PYR_LEVELS];
		float* pc_idepth[PYR_LEVELS];
		float* pc_color[PYR_LEVELS];

		// 跟踪时特征在关键帧上的信息
		float* buf_warped_idepth;
		float* buf_warped_u;
		float* buf_warped_v;
		float* buf_warped_dx;
		float* buf_warped_dy;
		float* buf_warped_residual;
		float* buf_warped_weight;
		float* buf_warped_refColor;
		int buf_warped_n;

		std::vector<float*> ptrToDelete;

		Accumulator9 acc;
	};

	class CoarseDistanceMap 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		CoarseDistanceMap(int w, int h);
		~CoarseDistanceMap();

		void makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian* frame);
		void makeInlierVotes(std::vector<FrameHessian*> frameHessians);

		void makeK(CalibHessian* HCalib);

		int* fwdWarpedIDDistFinal;

		// 各层金字塔相机内参信息
		Mat33f K[PYR_LEVELS];
		Mat33f Ki[PYR_LEVELS];
		float fx[PYR_LEVELS];
		float fy[PYR_LEVELS];
		float fxi[PYR_LEVELS];
		float fyi[PYR_LEVELS];
		float cx[PYR_LEVELS];
		float cy[PYR_LEVELS];
		float cxi[PYR_LEVELS];
		float cyi[PYR_LEVELS];

		// 各层金字塔图像宽高信息
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		void addIntoDistFinal(int u, int v);

	private:

		PointFrameResidual** coarseProjectionGrid;
		int* coarseProjectionGridNum;
		Eigen::Vector2i* bfsList1;
		Eigen::Vector2i* bfsList2;

		void growDistBFS(int bfsNum);
	};
}