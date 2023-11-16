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
	/// ɾ��ָ������v�еĵ�i��Ԫ��
	/// </summary>
	/// <typeparam name="T">ָ������</typeparam>
	/// <param name="v">ָ������</param>
	/// <param name="i">��ɾ��Ԫ������</param>
	template<typename T>
	inline void deleteOut(std::vector<T*>& v, const int i)
	{
		if (v[i] != NULL)
			delete v[i];
		v[i] = v.back();
		v.pop_back();
	}

	/// <summary>
	/// ɾ��ָ������v�е�ַΪi��Ԫ��
	/// </summary>
	/// <typeparam name="T">ָ������</typeparam>
	/// <param name="v">ָ������</param>
	/// <param name="i">��ɾ��Ԫ�ص�ַ</param>
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
	/// ɾ��ָ������v�е�i��Ԫ�أ����Ѻ����Ԫ����ǰ�ƶ�
	/// </summary>
	/// <typeparam name="T">ָ������</typeparam>
	/// <param name="v">ָ������</param>
	/// <param name="i">��ɾ��Ԫ������</param>
	template<typename T>
	inline void deleteOutOrder(std::vector<T*>& v, const int i)
	{
		if (v[i] != NULL) delete v[i];
		for (unsigned int k = i + 1; k < v.size(); k++)
			v[k - 1] = v[k];
		v.pop_back();
	}

	/// <summary>
	/// ɾ��ָ������v�е�ַΪelement��Ԫ�أ����Ѻ����Ԫ����ǰ�ƶ�
	/// </summary>
	/// <typeparam name="T">ָ������</typeparam>
	/// <param name="v">ָ������</param>
	/// <param name="element">��ɾ��Ԫ�ص�ַ</param>
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
	/// �����������Ƿ����nan��Ԫ��
	/// </summary>
	/// <param name="m">����������</param>
	/// <param name="msg">������Ϣ</param>
	/// <returns>�������Ƿ���nanԪ��</returns>
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
		/// ��Ե��ָ�������ؼ�֡������Ϊ:
		/// 1������Ż���ʹ���������Ե���ùؼ�֡;
		/// 2��ǰ�˺���ж�ɾ��Ŀ��֡Ϊ��Ե��֡�Ĺ۲�в�;
		/// 3������Ե���Ĺؼ�֡���͵���ʾ�߳���ʾ������¼��֡��ͳ����Ϣ;
		/// 4��ǰ�˹���Ļ����ؼ�֡��ɾ����Ե��֡�������ùؼ�֡��ţ����¼���ؼ�֡��Թ�ϵ;
		/// </summary>
		/// <param name="frame">����Ե���Ĺؼ�֡</param>
		void marginalizeFrame(FrameHessian* frame);

		/// <summary>
		/// ���������̣߳����̣߳����ȴ���ͼ�߳��������
		/// </summary>
		void blockUntilMappingIsFinished();

		/// <summary>
		/// ��˻����Ż�������
		/// </summary>
		/// <param name="numOptIts">�Ż���������</param>
		/// <returns>�Ż���Ĺ�Ȳв�</returns>
		float optimize(int numOptIts);

		/// <summary>
		/// ���ϵͳ������֡��λ����Ϣ
		/// </summary>
		/// <param name="file">��Ϣ����·��</param>
		void printResult(std::string file);

		void debugPlot(std::string name);

		/// <summary>
		/// ��ӡϵͳ������֡��¼��ͳ����Ϣ��������ʱ��Ե������Ч�в�������
		/// </summary>
		void printFrameLifetimes();

		/// <summary>
		/// ����ͼ��GammaУ������
		/// </summary>
		/// <param name="BInv">У������</param>
		void setGammaFunction(float* BInv);

		/// <summary>
		/// ����ϵͳ���ٵ�֡λ����Ϣ
		/// </summary>
		/// <param name="T">λ����Ϣ</param>
		void savetrajectory(const Sophus::Matrix4d& T);

		/// <summary>
		/// ����TUM��ʽ����ϵͳ���ٵõ���λ����Ϣ
		/// </summary>
		/// <param name="T">λ����Ϣ</param>
		/// <param name="time">ʱ�����Ϣ</param>
		void savetrajectoryTum(const SE3& T, const double time);

		/// <summary>
		/// ��ʼ���ߵ���Ϣ�����õ�һ֡����ϵΪ����ϵ�Լ���������ϵ��DSOϵ֮��ı任
		/// </summary>
		/// <param name="fh">����ϵͳ�ĵ�һ֡����</param>
		void initFirstFrameImu(FrameHessian* fh);

	public:
		int newFrameID;				// ϵͳ���½����֡��ͼ�������е�ID
		bool isLost;				// ϵͳ���ٶ�ʧ�ı�־λ
		bool initFailed;			// ϵͳ��ʼ��ʧ�ܵı�־λ
		bool initialized;			// ϵͳ�Ƿ�ɹ���ʼ���ı�־λ
		bool linearizeOperation;	// ���ٳɹ�����н�ͼ����ʱ�Ƿ������֡����ģʽ

		std::vector<IOWrap::Output3DWrapper*> outputWrapper;	// ��ʾ�߳�handle

	private:
		// FullSystemOptPoint
		PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

		// FullSystem mainPipelineFunctions
		void printLogLine();
		void printEigenValLine();
		Vec4 trackNewCoarse(FrameHessian* fh);

		/// <summary>
		/// �����ؼ�֡�е�δ�����������֡fh�и���һ�飬�Ż��������Ϣ
		/// </summary>
		/// <param name="fh">ϵͳ�е�����֡</param>
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

		// ����ģ���log��Ϣ
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

		// ����ģ���ͳ����Ϣ
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
		std::vector<FrameShell*> allFrameHistory;		// ϵͳ������֡��shell��Ϣ
		CoarseInitializer* coarseInitializer;			// ϵͳ��ʼ��handle
		Vec5 lastCoarseRMSE;							// ϵͳλ�˸�������һ���ٹ���ÿ���������Ȳв�

		// ================== changed by mapper-thread. protected by mapMutex ===============
		std::mutex mapMutex;							// ���ȫ�ֵ�ͼ���õĻ�����
		std::vector<FrameShell*> allKeyFramesHistory;	// ��¼ϵͳ��ɸѡ�������йؼ�֡

		EnergyFunctional* ef;							// �����Ż�handle
		IndexThreadReduce<Vec10> treadReduce;			// ���߳�handle

		float* selectionMap;
		PixelSelector* pixelSelector;
		CoarseDistanceMap* coarseDistanceMap;

		std::vector<FrameHessian*> frameHessians;		// ά��ϵͳ�����Ż�ʱ�����еĹؼ�֡
		std::vector<FrameHessian*> frameHessiansRight;	// ά��ϵͳ�����Ż�ʱ�����еĹؼ�֡�����˫Ŀ�����ͼ��
		std::vector<PointFrameResidual*> activeResiduals;
		float currentMinActDist;

		std::vector<float> allResVec;

		// �����н����һ֡��ΪcoarseTracker�ο��ؼ�֡����ͼ�̻߳�ı们���ؼ�֡���У���ʱ�����߳�Ӧ��
		// ���¸���coarseTracker�ο��ؼ�֡��ϵͳ����������coarseTracker handle�ֱ����ο�֡�ؼ�֡�仯
		// ǰ���coarseTracker
		std::mutex coarseTrackerSwapMutex;				// ��ͼ�̺߳͸����̶߳��������������handle����ӻ���������
		CoarseTracker* coarseTracker_forNewKF;			// tracker�ο��ؼ�֡�仯��ʵ�ʲ��õĸ���handle
		CoarseTracker* coarseTracker;					// tracker�ο��ؼ�֡�仯ǰʵ�ʲ��õĸ���handle
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
		std::mutex trackMapSyncMutex;								// �����̺߳ͽ�ͼ�߳�����ͬ���Ļ�����
		std::condition_variable trackedFrameSignal;					// �����̸߳�֪��ͼ�߳̿�ʼ��ͼ��������������
		std::condition_variable mappedFrameSignal;					// ��ͼ�̸߳�֪�����߳̽�ͼ�Ѿ���ɵ���������
		std::deque<FrameHessian*> unmappedTrackedFrames;			// �����̴߳��ݣ���ͼ�̴߳���ɽ�ͼ������֡���ݣ���Ŀ��
		std::deque<FrameHessian*> unmappedTrackedFramesRight;		// �����̴߳��ݣ���ͼ�̴߳���ɽ�ͼ������֡���ݣ���Ŀ��
		int needNewKFAfter;						// ����֡Ϊ�ؼ�֡������Ϊ����֡�Ĳο��ؼ�֡ID������ڵ��ڻ��������һ֡��ID
		std::thread mappingThread;				// ��ͼ�߳�handle
		bool runMapping;						// ���ý�ͼ�߳��Ƿ�����
		bool needToKetchupMapping;

		int lastRefStopID;
	};
}