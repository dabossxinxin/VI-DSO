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
	template<typename T>
	inline void deleteOut(std::vector<T*>& v, const int i)
	{
		delete v[i];
		v[i] = v.back();
		v.pop_back();
	}

	template<typename T>
	inline void deleteOutPt(std::vector<T*>& v, const T* i)
	{
		delete i;

		for (unsigned int k = 0; k < v.size(); k++)
		{
			if (v[k] == i)
			{
				v[k] = v.back();
				v.pop_back();
			}
		}
	}

	template<typename T>
	inline void deleteOutOrder(std::vector<T*>& v, const int i)
	{
		delete v[i];
		for (unsigned int k = i + 1; k < v.size(); k++)
			v[k - 1] = v[k];
		v.pop_back();
	}

	template<typename T>
	inline void deleteOutOrder(std::vector<T*>& v, const T* element)
	{
		int i = -1;
		for (unsigned int k = 0; k < v.size(); k++)
		{
			if (v[k] == element)
			{
				i = k;
				break;
			}
		}
		assert(i != -1);

		for (unsigned int k = i + 1; k < v.size(); k++)
			v[k - 1] = v[k];
		v.pop_back();

		delete element;
	}

	inline bool eigenTestNan(const MatXX& m, std::string msg)
	{
		bool foundNan = false;
		for (int y = 0; y < m.rows(); y++)
		{
			for (int x = 0; x < m.cols(); x++)
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
		void addActiveFrame(ImageAndExposure* image, ImageAndExposure* image_right, int id);

		// marginalizes a frame. drops / marginalizes points & residuals.
		void marginalizeFrame(FrameHessian* frame);
		void blockUntilMappingIsFinished();

		float optimize(int mnumOptIts);

		void printResult(std::string file);

		void debugPlot(std::string name);

		void printFrameLifetimes();
		// contains pointers to active frames

		std::vector<IOWrap::Output3DWrapper*> outputWrapper;

		bool isLost;				// 系统跟踪丢失的标志位
		bool initFailed;			// 系统初始化失败的标志位
		bool initialized;			// 系统是否成功初始化的标志位
		bool linearizeOperation;	// 是否通过

		void setGammaFunction(float* BInv);
		void setOriginalCalib(const VecXf& originalCalib, int originalW, int originalH);

		void savetrajectory(const Sophus::Matrix4d& T);
		void savetrajectory_tum(const SE3& T, double time);

		void initFirstFrame_imu(FrameHessian* fh);

	private:

		CalibHessian Hcalib;

		// opt single point
		int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
		PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

		double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

		// mainPipelineFunctions
		Vec4 trackNewCoarse(FrameHessian* fh);
		void traceNewCoarse(FrameHessian* fh);
		void traceNewCoarseNonKey(FrameHessian* fh, FrameHessian* fhRight);
		void traceNewCoarseKey(FrameHessian* fh, FrameHessian* fhRight);
		void activatePoints();
		void activatePointsMT();
		void activatePointsOldFirst();
		void flagPointsForRemoval();
		void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
		void initializeFromInitializer(FrameHessian* newFrame);
		void flagFramesForMarginalization(FrameHessian* newFH);

		void removeOutliers();

		// 计算滑窗关键帧之间的相互关系
		void setPrecalcValues();

		// solce. eventually migrate to ef.
		void solveSystem(int iteration, double lambda);
		Vec3 linearizeAll(bool fixLinearization);
		bool doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD);
		void backupState(bool backupLastStep);
		void loadSateBackup();
		double calcLEnergy();
		double calcMEnergy();
		void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
		void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized, std::vector<ImmaturePoint*>* toOptimize, int min, int max, Vec10* stats, int tid);
		void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);

		void printOptRes(const Vec3& res, double resL, double resM, double resPrior, double LExact, float a, float b);

		void debugPlotTracking();

		std::vector<VecX> getNullspaces(
			std::vector<VecX>& nullspaces_pose,
			std::vector<VecX>& nullspaces_scale,
			std::vector<VecX>& nullspaces_affA,
			std::vector<VecX>& nullspaces_affB);

		void setNewFrameEnergyTH();

		void printLogLine();
		void printEvalLine();
		void printEigenValLine();

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
		std::vector<FrameHessian*> frameHessians_right;	// 维护系统滑窗优化时滑窗中的关键帧，针对双目右相机图像
		std::vector<PointFrameResidual*> activeResiduals;
		float currentMinActDist;

		std::vector<float> allResVec;

		// mutex etc. for tracker exchange.
		std::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
		CoarseTracker* coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
		CoarseTracker* coarseTracker;					// always used to track new frames. protected by [trackMutex].
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