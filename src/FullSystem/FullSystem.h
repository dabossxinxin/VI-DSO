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

#define MAX_ACTIVE_FRAMES 100

#include <iostream>
#include <fstream>
#include <deque>
#include <math.h>
#include <vector>

#include "util/NumType.h"
#include "util/FrameShell.h"
#include "util/globalCalib.h"
#include "util/IndexThreadReduce.h"
#include "util/ImageAndExposure.h"

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/HessianBlocks.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{
	/// <summary>
	/// 删除指针序列v中的第i个元素
	/// </summary>
	/// <typeparam name="T">指针类型</typeparam>
	/// <param name="v">指针序列</param>
	/// <param name="i">待删除元素索引</param>
	template<typename T>
	inline void deleteOut(std::vector<T*>& v, const int i)
	{
		if (v[i] != NULL)
			delete v[i];
		v[i] = v.back();
		v.pop_back();
	}

	/// <summary>
	/// 删除指针序列v中地址为i的元素
	/// </summary>
	/// <typeparam name="T">指针类型</typeparam>
	/// <param name="v">指针序列</param>
	/// <param name="i">待删除元素地址</param>
	template<typename T>
	inline void deleteOutPt(std::vector<T*>& v, const T* i)
	{
		if (i != NULL) delete i;
		for (unsigned int k = 0; k < v.size(); ++k)
		{
			if (v[k] == i)
			{
				v[k] = v.back();
				v.pop_back();
			}
		}
	}

	/// <summary>
	/// 删除指针序列v中第i个元素，并把后面的元素往前移动
	/// </summary>
	/// <typeparam name="T">指针类型</typeparam>
	/// <param name="v">指针序列</param>
	/// <param name="i">待删除元素索引</param>
	template<typename T>
	inline void deleteOutOrder(std::vector<T*>& v, const int i)
	{
		if (v[i] != NULL) delete v[i];
		for (unsigned int k = i + 1; k < v.size(); k++)
			v[k - 1] = v[k];
		v.pop_back();
	}

	/// <summary>
	/// 删除指针序列v中地址为element的元素，并把后面的元素往前移动
	/// </summary>
	/// <typeparam name="T">指针类型</typeparam>
	/// <param name="v">指针序列</param>
	/// <param name="element">待删除元素地址</param>
	template<typename T>
	inline void deleteOutOrder(std::vector<T*>& v, const T* element)
	{
		int idx = -1;
		for (unsigned int k = 0; k < v.size(); ++k)
		{
			if (v[k] == element)
			{
				idx = k;
				break;
			}
		}
		assert(idx != -1);

		for (unsigned int k = idx + 1; k < v.size(); k++)
			v[k - 1] = v[k];
		v.pop_back();
		SAFE_DELETE(element);
	}

	/// <summary>
	/// 搜索矩阵中是否存在nan的元素
	/// </summary>
	/// <param name="m">待搜索矩阵</param>
	/// <param name="msg">调试信息</param>
	/// <returns>矩阵中是否有nan元素</returns>
	inline bool eigenTestNan(const MatXX& m, std::string msg)
	{
		bool foundNan = false;
		for (int y = 0; y < m.rows(); ++y)
		{
			for (int x = 0; x < m.cols(); ++x)
			{
				if (!std::isfinite((double)m(y, x))) foundNan = true;
			}
		}

		if (foundNan)
		{
			printf("NAN in %s:\n", msg.c_str());
			std::cout << m << "\n\n";
		}

		return foundNan;
	}

	class FullSystem
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		FullSystem();
		virtual ~FullSystem();

		// adds a new frame, and creates point & residual structs.
		void addActiveFrame(ImageAndExposure* image, ImageAndExposure* imageRight, int id);

		/// <summary>
		/// 边缘化指定滑窗关键帧，步骤为:
		/// 1、后端优化中使用舒尔补边缘化该关键帧;
		/// 2、前端后端中都删除目标帧为边缘化帧的观测残差;
		/// 3、将边缘化的关键帧发送到显示线程显示，并记录该帧的统计信息;
		/// 4、前端管理的滑窗关键帧中删除边缘化帧，并重置关键帧序号，重新计算关键帧相对关系;
		/// </summary>
		/// <param name="frame">待边缘化的关键帧</param>
		void marginalizeFrame(FrameHessian* frame);

		/// <summary>
		/// 阻塞跟踪线程（主线程），等待建图线程完成任务
		/// </summary>
		void blockUntilMappingIsFinished();

		/// <summary>
		/// 后端滑窗优化主函数
		/// </summary>
		/// <param name="numOptIts">优化迭代次数</param>
		/// <returns>优化后的光度残差</returns>
		float optimize(int numOptIts);

		/// <summary>
		/// 输出系统中所有帧的位姿信息
		/// </summary>
		/// <param name="file">信息保存路径</param>
		void printResult(std::string file);

		void debugPlot(std::string name);

		/// <summary>
		/// 打印系统中所有帧记录的统计信息，包含何时边缘化，有效残差数量等
		/// </summary>
		void printFrameLifetimes();

		/// <summary>
		/// 设置图像Gamma校正参数
		/// </summary>
		/// <param name="BInv">校正参数</param>
		void setGammaFunction(float* BInv);

		/// <summary>
		/// 保存系统跟踪的帧位姿信息
		/// </summary>
		/// <param name="T">位姿信息</param>
		void savetrajectory(const Sophus::Matrix4d& T);

		/// <summary>
		/// 按照TUM格式保存系统跟踪得到的位姿信息
		/// </summary>
		/// <param name="T">位姿信息</param>
		/// <param name="time">时间戳信息</param>
		void savetrajectoryTum(const SE3& T, const double time);

		/// <summary>
		/// 初始化惯导信息：设置第一帧坐标系为世界系以及定义世界系与DSO系之间的变换
		/// </summary>
		/// <param name="fh">进入系统的第一帧数据</param>
		void initFirstFrameImu(FrameHessian* fh);

	public:
		int newFrameID;				// 系统最新进入的帧在图像序列中的ID
		bool isLost;				// 系统跟踪丢失的标志位
		bool initFailed;			// 系统初始化失败的标志位
		bool initialized;			// 系统是否成功初始化的标志位
		bool linearizeOperation;	// 跟踪成功后进行建图操作时是否进行逐帧操作模式

		std::vector<IOWrap::Output3DWrapper*> outputWrapper;	// 显示线程handle

	private:
		// FullSystemOptPoint
		PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

		// FullSystem mainPipelineFunctions
		void printLogLine();
		void printEigenValLine();
		Vec4 trackNewCoarse(FrameHessian* fh);

		/// <summary>
		/// 滑窗关键帧中的未激活点在最新帧fh中跟踪一遍，优化其深度信息
		/// </summary>
		/// <param name="fh">系统中的最新帧</param>
		void traceNewCoarse(FrameHessian* fh);

		void traceNewCoarseNonKey(FrameHessian* fh, FrameHessian* fhRight);
		void traceNewCoarseKey(FrameHessian* fh, FrameHessian* fhRight);
		void setPrecalcValues();
		void activatePointsMT();
		void flagPointsForRemoval();
		void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
		void initializeFromInitializer(FrameHessian* newFrame);
		void flagFramesForMarginalization(FrameHessian* newFH);
		void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,
			std::vector<ImmaturePoint*>* toOptimize, int min, int max, Vec10* stats, int tid);

		// FullSystemOptimize
		void removeOutliers();
		void loadSateBackup();
		double calcLEnergy();
		double calcMEnergy();
		void setNewFrameEnergyTH();
		void backupState(bool backupLastStep);
		Vec3 linearizeAll(bool fixLinearization);
		void solveSystem(int iteration, double lambda);
		bool doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD);
		void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
		void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
		void printOptRes(const Vec3& res, double resL, double resM, double resPrior, double LExact, float a, float b);
		std::vector<VecX> getNullspaces(std::vector<VecX>& nullspaces_pose, std::vector<VecX>& nullspaces_scale,
			std::vector<VecX>& nullspaces_affA, std::vector<VecX>& nullspaces_affB);
		void debugPlotTracking();

		CalibHessian Hcalib;

		// 各个模块的log信息
		std::ofstream* calibLog;
		std::ofstream* numsLog;
		std::ofstream* errorsLog;
		std::ofstream* eigenAllLog;
		std::ofstream* eigenPLog;
		std::ofstream* eigenALog;
		std::ofstream* DiagonalLog;
		std::ofstream* variancesLog;
		std::ofstream* nullspacesLog;
		std::ofstream* coarseTrackingLog;

		// 各个模块的统计信息
		long int statistics_lastNumOptIts;
		long int statistics_numDroppedPoints;
		long int statistics_numActivatedPoints;
		long int statistics_numCreatedPoints;
		long int statistics_numForceDroppedResBwd;
		long int statistics_numForceDroppedResFwd;
		long int statistics_numMargResFwd;
		long int statistics_numMargResBwd;
		float statistics_lastFineTrackRMSE;

		// =================== changed by tracker-thread. protected by trackMutex ============
		std::mutex trackMutex;
		std::vector<FrameShell*> allFrameHistory;		// 系统中所有帧的shell信息
		CoarseInitializer* coarseInitializer;			// 系统初始化handle
		Vec5 lastCoarseRMSE;							// 系统位姿跟踪中上一跟踪过程每层金字塔光度残差

		// ================== changed by mapper-thread. protected by mapMutex ===============
		std::mutex mapMutex;							// 针对全局地图设置的互斥锁
		std::vector<FrameShell*> allKeyFramesHistory;	// 记录系统中筛选出的所有关键帧

		EnergyFunctional* ef;							// 滑窗优化handle
		IndexThreadReduce<Vec10> treadReduce;			// 多线程handle

		float* selectionMap;
		PixelSelector* pixelSelector;
		CoarseDistanceMap* coarseDistanceMap;

		std::vector<FrameHessian*> frameHessians;		// 维护系统滑窗优化时滑窗中的关键帧
		std::vector<FrameHessian*> frameHessiansRight;	// 维护系统滑窗优化时滑窗中的关键帧，针对双目右相机图像
		std::vector<PointFrameResidual*> activeResiduals;
		float currentMinActDist;

		std::vector<float> allResVec;

		// 滑窗中将最后一帧作为coarseTracker参考关键帧，建图线程会改变滑窗关键帧序列，此时跟踪线程应该
		// 重新更新coarseTracker参考关键帧，系统中设置两个coarseTracker handle分别管理参考帧关键帧变化
		// 前后的coarseTracker
		std::mutex coarseTrackerSwapMutex;				// 建图线程和跟踪线程都会访问下面两个handle，添加互斥锁保护
		CoarseTracker* coarseTracker_forNewKF;			// tracker参考关键帧变化后实际采用的跟踪handle
		CoarseTracker* coarseTracker;					// tracker参考关键帧变化前实际采用的跟踪handle
		float minIdJetVisTracker, maxIdJetVisTracker;
		float minIdJetVisDebug, maxIdJetVisDebug;

		// mutex for camToWorl's in shells (these are always in a good configuration).
		std::mutex shellPoseMutex;

		// tracking always uses the newest KF as reference
		void makeKeyFrame(FrameHessian* fh, FrameHessian* fhRight);
		void makeNonKeyFrame(FrameHessian* fh, FrameHessian* fhRight);
		void deliverTrackedFrame(FrameHessian* fh, FrameHessian* fhRight, bool needKF);
		void mappingLoop();

		// tracking / mapping synchronization. All protected by [trackMapSyncMutex].
		std::mutex trackMapSyncMutex;								// 跟踪线程和建图线程数据同步的互斥锁
		std::condition_variable trackedFrameSignal;					// 跟踪线程告知建图线程开始建图操作的条件变量
		std::condition_variable mappedFrameSignal;					// 建图线程告知跟踪线程建图已经完成的条件变量
		std::deque<FrameHessian*> unmappedTrackedFrames;			// 跟踪线程传递，建图线程待完成建图操作的帧数据（左目）
		std::deque<FrameHessian*> unmappedTrackedFramesRight;		// 跟踪线程传递，建图线程待完成建图操作的帧数据（右目）
		int needNewKFAfter;						// 最新帧为关键帧的条件为最新帧的参考关键帧ID必须大于等于滑窗中最后一帧的ID
		std::thread mappingThread;				// 建图线程handle
		bool runMapping;						// 设置建图线程是否运行
		bool needToKetchupMapping;

		int lastRefStopID;
	};
}