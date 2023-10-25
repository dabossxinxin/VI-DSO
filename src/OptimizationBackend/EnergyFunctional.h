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
#include "util/threading.h"

#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/IMUPreintegrator.h"

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

namespace dso
{
	//class AccumulatedTopHessian;
	//class AccumulatedSCHessian;

	extern bool EFAdjointsValid;		// �Ż������в���������Ծ��������ſɱ��Ƿ�������
	extern bool EFIndicesValid;			// �Ż������л����ؼ�֡�Լ���������������Ƿ���
	extern bool EFDeltaValid;			// �Ż������л����ؼ�֡�Լ����������������Ƿ����ú�

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

		// �����Ż�����
		void solveSystemF(int iteration, double lambda, CalibHessian* HCalib);
		double calcMEnergyF();
		double calcLEnergyF_MT();

		void makeIDX();

		void setDeltaF(CalibHessian* HCalib);

		void setAdjointsF(CalibHessian* Hcalib);

		std::vector<EFFrame*> frames;		// ���������еĹؼ�֡
		int nPoints;						// ���뻬���Ż�����������
		int nFrames;						// ���뻬���Ż��Ĺؼ�֡����
		int nResiduals;						// ���뻬���Ż��Ĺ�Ȳв�����

		MatXX HM;							// �Ӿ����ֱ�Ե����H
		VecX bM;							// �Ӿ����ֱ�Ե����b
		    
		MatXX HM_imu;						// �ߵ����ֱ�Ե����H
		VecX bM_imu;						// �ߵ����ֱ�Ե����b

		MatXX HM_bias;
		VecX bM_bias;

		MatXX HM_imu_half;
		VecX bM_imu_half;

		double s_middle = 1;
		double s_last = 1;
		double d_now = sqrt(1.1);
		double d_half = sqrt(1.1);
		bool side_last = true;		//for dynamic margin: true: upper s_middle false: below s_middle

		int resInA, resInL, resInM;
		MatXX lastHS;
		VecX lastbS;
		VecX lastX;
		std::vector<VecX> lastNullspaces_forLogging;
		std::vector<VecX> lastNullspaces_pose;
		std::vector<VecX> lastNullspaces_scale;
		std::vector<VecX> lastNullspaces_affA;
		std::vector<VecX> lastNullspaces_affB;

		//ThreadPool* threadPool;
		IndexThreadReduce<Vec10>* red;

		std::map<uint64_t,
			Eigen::Vector2i,
			std::less<uint64_t>,
			Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
		> connectivityMap;
	private:

		VecX getStitchedDeltaF() const;

		// �������õ������λ���Լ�����ڲθ�������������������ȵĸ����� 
		void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
		void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

		void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

		void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

		void calcIMUHessian(MatXX &H, VecX &b);

		void orthogonalize(VecX* b, MatXX* H);
		Mat18f* adHTdeltaF;		// hostToTargetλ�ˡ���ȱ任���� [0~5�����λ�ˣ�6��7����Ȳ���]

		Mat88* adHost;			// hostToTargetλ�ˡ���ȱ任��hostλ�ˡ���ȵ��ſɱ� [double]
		Mat88* adTarget;		// hostToTargetλ�ˡ���ȱ任��targetλ�ˡ���ȵ��ſɱ� [double]
		Mat88f* adHostF;		// hostToTargetλ�ˡ���ȱ任��hostλ�ˡ���ȵ��ſɱ� [float]
		Mat88f* adTargetF;		// hostToTargetλ�ˡ���ȱ任��targetλ�ˡ���ȵ��ſɱ� [float]

		VecCf cDeltaF;			// ����ڲβ�������
		VecC cPrior;			// ����ڲβ�������ֵ [double]
		VecCf cPriorF;			// ����ڲβ�������ֵ [float]
		
		AccumulatedTopHessianSSE* accSSE_top_L;		// ���㻬���м���������Heesian��Ϣ
		AccumulatedTopHessianSSE* accSSE_top_A;		// ���㻬�������Ի�������Hessian��Ϣ
		AccumulatedSCHessianSSE* accSSE_bot;		// ���㻬���б�Ե���������������Ϣ

		std::vector<EFPoint*> allPoints;			// �����Ż��������������
		std::vector<EFPoint*> allPointsToMarg;		// �����Ż���������б�Ե������

		float currentLambda;
	};
}