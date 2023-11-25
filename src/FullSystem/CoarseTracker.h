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
		/// ��������֡��λ�˺͹�Ȳ���
		/// </summary>
		/// <param name="newFrameHessian">��������֡</param>
		/// <param name="lastToNew_out">������һ�ؼ�֡������֡��λ�˱任</param>
		/// <param name="aff_g2l_out">�����Ȳ����仯</param>
		/// <param name="coarsestLvl">�ܽ���������</param>
		/// <param name="minResForAbort">ÿ����������õĹ�Ȳв���ֵ</param>
		/// <param name="wrap">��ʾ�߳�handle</param>
		/// <returns>�Ƿ�ɹ���������֡�ı�ֵ</returns>
		bool trackNewestCoarse(FrameHessian* newFrameHessian, SE3& lastToNew_out, AffLight& aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort, IOWrap::Output3DWrapper* wrap = nullptr);

		void setCTRefForFirstFrame(std::vector<FrameHessian*> frameHessians);
		void setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians, FrameHessian* fhRight, CalibHessian Hcalib);
		
		void makeCoarseDepthForFirstFrame(FrameHessian* fh);
		void makeK(CalibHessian* HCalib);

		bool debugPrint;		// ���Ƶ�����Ϣ����ı�־λ
		bool debugPlot;			// ���Ƶ�����Ϣ���Ƶı�־λ

		// ���������㼶�ڲ���Ϣ
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

		// ��������ͼ����
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		/// <summary>
		/// �������ؼ�֡������������ο��ؼ�֡ͶӰ�����ƶ�Ӧ�������ͼ
		/// </summary>
		/// <param name="minID_pt">��С�����</param>
		/// <param name="maxID_pt">��������</param>
		/// <param name="wraps">��ʾ�߳�handle</param>
		void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);

		void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

		FrameHessian* lastRef;			// ����ʱ�Ĳο�֡
		AffLight lastRef_aff_g2l;		// ����ʱ�ο�֡�Ĺ�ȱ任����
		FrameHessian* newFrame;			// �����ٵ�����֡
		int refFrameID;					// ����ʱ�ο�֡��ID

		Vec5 lastResiduals;				// ����������в���Ϣ
		Vec3 lastFlowIndicators;		// ����������������Ϣ
		double firstCoarseRMSE;			// �״θ��ٵ�����������Ȳв�
		int pc_n[PYR_LEVELS];
	private:

		void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fhrRight, CalibHessian Hcalib);
		
		Vec7 calcIMUResAndGS(Mat66 &H_out, Vec6 &b_out, SE3 &refToNew, const IMUPreintegrator &IMU_preintegrator, const int lvl, double trackWeight);
		Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

		// ������������ο��ؼ�֡lastRef��ͶӰ�������ֲ�ͬ����ͶӰ��ͬһ�������ϵ������
		// ��ʱ��¼ÿһ��ͶӰ������Heesian����ΪȨ�����ڼ�����������ȵĹ�һ����
		float* idepth[PYR_LEVELS];				// ���������㼶���������
		float* weightSums[PYR_LEVELS];			// ���������㼶���������Ȩ��
		float* weightSums_bak[PYR_LEVELS];		// ���������㼶���������Ȩ�ر���

		// ����ʱ�����ڲο�֡�ϵ���Ϣ
		float* pc_u[PYR_LEVELS];
		float* pc_v[PYR_LEVELS];
		float* pc_idepth[PYR_LEVELS];
		float* pc_color[PYR_LEVELS];

		// ����ʱ�����ڹؼ�֡�ϵ���Ϣ
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
	/// �����ͼ������ͼ���и������ص����������ľ��룬���ڻ�������ʱѡ��
	/// </summary>
	class CoarseDistanceMap 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		/// <summary>
		/// CoarseDistanceMap���캯���������Ҫ�ڴ�ռ䣻
		/// ������Ҫ���ڽ�������һ�㹹������ͼ������ÿ�����ص㵽��������ľ���
		/// </summary>
		/// <param name="ww">ͼ����</param>
		/// <param name="hh">ͼ��߶�</param>
		CoarseDistanceMap(int w, int h);

		/// <summary>
		/// CoarseTracker�����������ͷų�Ա�ڴ�ռ�
		/// </summary>
		~CoarseDistanceMap();

		/// <summary>
		/// ��������ͼ�������Ǿ���ѡȡ��������δ����㼤�����
		/// </summary>
		/// <param name="frameHessians">���������йؼ�֡</param>
		/// <param name="frame">�����ؼ�֡�����һ֡�ؼ�֡</param>
		void makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian* frame);

		/// <summary>
		/// ���ƾ����ͼ��Ϊ�˽�Լʱ�䣬�����ͼʵ�����ڽ�������һ�����
		/// </summary>
		void debugPlotDistanceMap();

		/// <summary>
		/// ����DistanceMap�и������������ڲ���Ϣ
		/// </summary>
		/// <param name="HCalib">����ڲ���Ϣ</param>
		void makeK(CalibHessian* HCalib);

		unsigned char* fwdWarpedIDDistFinal;
		MinimalImageB* debugImage;

		// �������������ڲ���Ϣ
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

		// ���������ͼ������Ϣ
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		/// <summary>
		/// ������ͼ����������������µľ����ͼ
		/// </summary>
		/// <param name="u">��������������u</param>
		/// <param name="v">��������������v</param>
		void addIntoDistFinal(int u, int v);

	private:

		PointFrameResidual** coarseProjectionGrid;
		int* coarseProjectionGridNum;
		Eigen::Vector2i* bfsList1;
		Eigen::Vector2i* bfsList2;

		/// <summary>
		/// ������ȱ��������һ���������ÿ�����ؾ���������������ؾ���
		/// </summary>
		/// <param name="bfsNum">������������</param>
		void growDistBFS(int bfsNum);
	};
}