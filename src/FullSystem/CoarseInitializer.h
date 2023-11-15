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
#include "util/settings.h"

#include "FullSystem/HessianBlocks.h"

#include "IOWrapper/Output3DWrapper.h"

#include "OptimizationBackend/MatrixAccumulators.h"

namespace dso
{
	struct Pnt
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		float u, v;			// 特征点的像素坐标值

		// idepth / isgood / energy during optimization.
		float idepth;		// 特征点逆深度值
		bool isGood;		// 特征点质量是否OK
		Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
		bool isGood_new;
		float idepth_new;
		Vec2f energy_new;

		float iR;			// 特征点逆深度均值
		float iRSumNum;

		float lastHessian;				// 特征点残差对逆深度雅可比的平方项
		float lastHessian_new;			// 特征点残差对逆深度雅可比的平方项

		// max stepsize for idepth (corresponding to max. movement in pixel-space).
		float maxstep;					// 特征优化迭代中允许的最大增量

		// idx (x+y*w) of closest point one pyramid level above.
		int parent;						// 该点在上一层金字塔中的最邻近点
		float parentDist;				// 该点与上一层金字塔中最邻近点的距离

		// idx (x+y*w) of up to 10 nearest points in pixel space.
		int neighbours[10];				// 同层金字塔中当前点的十个最邻近点
		float neighboursDist[10];		// 同层金字塔中当前点的十个最邻近点的距离

		float my_type;
		float outlierTH;				// 特征点光度能量阈值，大于该阈值特征
	};

	class CoarseInitializer 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		CoarseInitializer(int w, int h);
		~CoarseInitializer();

		// 将进入系统的第一帧图像设置为初始化器的第一帧
		void setFirst(CalibHessian* HCalib, FrameHessian* newFrame);
		void setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessianRight);

		// 跟踪系统最新送进来的图像帧并返回跟踪状态
		bool trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*>& wraps);
		void calcTGrads(FrameHessian* newFrameHessian);

		int frameID;						// 初始化过程中进入系统的图像帧数量	
		bool printDebug;					// DSO初始化中是否打印调试信息

		Pnt* points[PYR_LEVELS];			// 金字塔中选取的所有特征点的坐标
		int numPoints[PYR_LEVELS];			// 每一层金字塔中选取的特征点的数量
		AffLight thisToNext_aff;			// 初始化得到的参考帧到最新帧的光度变换
		SE3 thisToNext;						// 初始化得到的参考帧到最新帧的姿态

		FrameHessian* firstFrame;			// 初始化器中维护的最初进入系统的一帧
		FrameHessian* firstFrameRight;		// 最初进入系统的帧对应的双目右相机图像帧
		FrameHessian* newFrame;				// 最新进入初始化器中的帧用于与第一帧完成初始化

	private:

		Mat33 K[PYR_LEVELS];
		Mat33 Ki[PYR_LEVELS];
		double fx[PYR_LEVELS];
		double fy[PYR_LEVELS];
		double fxi[PYR_LEVELS];
		double fyi[PYR_LEVELS];
		double cx[PYR_LEVELS];
		double cy[PYR_LEVELS];
		double cxi[PYR_LEVELS];
		double cyi[PYR_LEVELS];
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		void makeK(CalibHessian* HCalib);

		float* idepth[PYR_LEVELS];

		bool snapped;				// 初始化中是否找到了位移较大的两帧
		int snappedAt;				// 初始化中在进入系统的第几帧图像中找到的位移较大的帧

		Eigen::Vector3f* dINew[PYR_LEVELS];		// 初始化中最新帧图像灰度以及梯度数据
		Eigen::Vector3f* dIFist[PYR_LEVELS];	// 初始化中第一帧图像灰度以及梯度数据

		Eigen::DiagonalMatrix<float, 8> wM;		// 降低Hessian矩阵条件数准备的权重矩阵

		Vec10f* JbBuffer;			// 计算舒尔补项的临时变量；0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
		Vec10f* JbBuffer_new;		// 计算舒尔补项的临时变量；0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.	

		Accumulator9 acc9;			// 计算光度残差对于相机位姿以及光度参数部分的Hessian
		Accumulator9 acc9SC;		// 计算光度残差对于特征逆深度部分的Hessian舒尔补项

		Vec3f dGrads[PYR_LEVELS];

		// 正则化相关参数
		float alphaK;
		float alphaW;
		float regWeight;
		float couplingWeight;

		Vec3f calcResAndGS(
			int lvl,
			Mat88f &H_out, Vec8f &b_out,
			Mat88f &H_out_sc, Vec8f &b_out_sc,
			const SE3 &refToNew, AffLight refToNew_aff,
			bool plot);

		Vec3f calcEC(int lvl); // returns OLD ENERGY, NEW ENERGY, NUM TERMS.
		void optReg(int lvl);

		void propagateUp(int srcLvl);
		void propagateDown(int srcLvl);
		float rescale();

		void resetPoints(int lvl);
		void doStep(int lvl, float lambda, Vec8f inc);
		void applyStep(int lvl);

		void makeGradients(Eigen::Vector3f** data); 
		void debugPlotDepth(int lvl, std::vector<IOWrap::Output3DWrapper*>& wraps);
		void makeNN();
	};

	struct FLANNPointcloud
	{
		inline FLANNPointcloud() { num = 0; points = 0; }
		inline FLANNPointcloud(int n, Pnt* p) : num(n), points(p) {}
		int num;
		Pnt* points;
		inline size_t kdtree_get_point_count() const { return num; }
		inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const
		{
			const float d0 = p1[0] - points[idx_p2].u;
			const float d1 = p1[1] - points[idx_p2].v;
			return d0 * d0 + d1 * d1;
		}

		inline float kdtree_get_pt(const size_t idx, int dim) const
		{
			if (dim == 0) return points[idx].u;
			else return points[idx].v;
		}
		template <class BBOX>
		bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
	};
}