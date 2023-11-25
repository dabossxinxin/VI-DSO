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

		/// <summary>
		/// 跟踪最新帧的位姿和光度参数
		/// </summary>
		/// <param name="newFrameHessian">输入最新帧</param>
		/// <param name="lastToNew_out">输入上一关键帧到最新帧的位姿变换</param>
		/// <param name="aff_g2l_out">输入光度参数变化</param>
		/// <param name="coarsestLvl">总金字塔层数</param>
		/// <param name="minResForAbort">每层金字塔设置的光度残差阈值</param>
		/// <param name="wrap">显示线程handle</param>
		/// <returns>是否成功跟踪最新帧的标值</returns>
		bool trackNewestCoarse(FrameHessian* newFrameHessian, SE3& lastToNew_out, AffLight& aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort, IOWrap::Output3DWrapper* wrap = nullptr);

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

		/// <summary>
		/// 将滑窗关键帧中特征数据向参考关键帧投影并绘制对应的逆深度图
		/// </summary>
		/// <param name="minID_pt">最小逆深度</param>
		/// <param name="maxID_pt">最大逆深度</param>
		/// <param name="wraps">显示线程handle</param>
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

		void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fhrRight, CalibHessian Hcalib);
		
		Vec7 calcIMUResAndGS(Mat66 &H_out, Vec6 &b_out, SE3 &refToNew, const IMUPreintegrator &IMU_preintegrator, const int lvl, double trackWeight);
		Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

		// 将所有特征向参考关键帧lastRef上投影，将出现不同特征投影到同一个像素上的情况，
		// 此时记录每一个投影特征的Heesian逆作为权重用于计算特征逆深度的归一化积
		float* idepth[PYR_LEVELS];				// 各金字塔层级特征逆深度
		float* weightSums[PYR_LEVELS];			// 各金字塔层级特征逆深度权重
		float* weightSums_bak[PYR_LEVELS];		// 各金字塔层级特征逆深度权重备份

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

	/// <summary>
	/// 距离地图：描述图像中各个像素到最近特征点的距离，便于滑窗跟踪时选点
	/// </summary>
	class CoarseDistanceMap 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		/// <summary>
		/// CoarseDistanceMap构造函数：分配必要内存空间；
		/// 该类主要是在金字塔第一层构造距离地图，描述每个像素点到最近特征的距离
		/// </summary>
		/// <param name="ww">图像宽度</param>
		/// <param name="hh">图像高度</param>
		CoarseDistanceMap(int w, int h);

		/// <summary>
		/// CoarseTracker析构函数，释放成员内存空间
		/// </summary>
		~CoarseDistanceMap();

		/// <summary>
		/// 构造距离地图，作用是均匀选取特征进行未成熟点激活操作
		/// </summary>
		/// <param name="frameHessians">滑窗中所有关键帧</param>
		/// <param name="frame">滑窗关键帧中最后一帧关键帧</param>
		void makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian* frame);

		/// <summary>
		/// 绘制距离地图：为了节约时间，距离地图实际上在金字塔第一层绘制
		/// </summary>
		void debugPlotDistanceMap();

		/// <summary>
		/// 设置DistanceMap中各层金字塔相机内参信息
		/// </summary>
		/// <param name="HCalib">相机内参信息</param>
		void makeK(CalibHessian* HCalib);

		unsigned char* fwdWarpedIDDistFinal;
		MinimalImageB* debugImage;

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

		/// <summary>
		/// 向距离地图中添加特征并构造新的距离地图
		/// </summary>
		/// <param name="u">新特征像素坐标u</param>
		/// <param name="v">新特征像素坐标v</param>
		void addIntoDistFinal(int u, int v);

	private:

		PointFrameResidual** coarseProjectionGrid;
		int* coarseProjectionGridNum;
		Eigen::Vector2i* bfsList1;
		Eigen::Vector2i* bfsList2;

		/// <summary>
		/// 广度优先遍历计算第一层金字塔中每个像素距离最近特征的像素距离
		/// </summary>
		/// <param name="bfsNum">输入特征数量</param>
		void growDistBFS(int bfsNum);
	};
}