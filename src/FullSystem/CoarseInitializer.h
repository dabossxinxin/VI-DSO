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

		float u, v;			// ���������������ֵ

		// idepth / isgood / energy during optimization.
		float idepth;		// �����������ֵ
		bool isGood;		// �����������Ƿ�OK
		Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
		bool isGood_new;
		float idepth_new;
		Vec2f energy_new;

		float iR;			// ����������Ⱦ�ֵ
		float iRSumNum;

		float lastHessian;				// ������в��������ſɱȵ�ƽ����
		float lastHessian_new;			// ������в��������ſɱȵ�ƽ����

		// max stepsize for idepth (corresponding to max. movement in pixel-space).
		float maxstep;					// �����Ż�������������������

		// idx (x+y*w) of closest point one pyramid level above.
		int parent;						// �õ�����һ��������е����ڽ���
		float parentDist;				// �õ�����һ������������ڽ���ľ���

		// idx (x+y*w) of up to 10 nearest points in pixel space.
		int neighbours[10];				// ͬ��������е�ǰ���ʮ�����ڽ���
		float neighboursDist[10];		// ͬ��������е�ǰ���ʮ�����ڽ���ľ���

		float my_type;
		float outlierTH;				// ��������������ֵ�����ڸ���ֵ����
	};

	class CoarseInitializer 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		CoarseInitializer(int w, int h);
		~CoarseInitializer();

		// ������ϵͳ�ĵ�һ֡ͼ������Ϊ��ʼ�����ĵ�һ֡
		void setFirst(CalibHessian* HCalib, FrameHessian* newFrame);
		void setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessianRight);

		// ����ϵͳ�����ͽ�����ͼ��֡�����ظ���״̬
		bool trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*>& wraps);
		void calcTGrads(FrameHessian* newFrameHessian);

		int frameID;						// ��ʼ�������н���ϵͳ��ͼ��֡����	
		bool printDebug;					// DSO��ʼ�����Ƿ��ӡ������Ϣ

		Pnt* points[PYR_LEVELS];			// ��������ѡȡ�����������������
		int numPoints[PYR_LEVELS];			// ÿһ���������ѡȡ�������������
		AffLight thisToNext_aff;			// ��ʼ���õ��Ĳο�֡������֡�Ĺ�ȱ任
		SE3 thisToNext;						// ��ʼ���õ��Ĳο�֡������֡����̬

		FrameHessian* firstFrame;			// ��ʼ������ά�����������ϵͳ��һ֡
		FrameHessian* firstFrameRight;		// �������ϵͳ��֡��Ӧ��˫Ŀ�����ͼ��֡
		FrameHessian* newFrame;				// ���½����ʼ�����е�֡�������һ֡��ɳ�ʼ��

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

		bool snapped;				// ��ʼ�����Ƿ��ҵ���λ�ƽϴ����֡
		int snappedAt;				// ��ʼ�����ڽ���ϵͳ�ĵڼ�֡ͼ�����ҵ���λ�ƽϴ��֡

		Eigen::Vector3f* dINew[PYR_LEVELS];		// ��ʼ��������֡ͼ��Ҷ��Լ��ݶ�����
		Eigen::Vector3f* dIFist[PYR_LEVELS];	// ��ʼ���е�һ֡ͼ��Ҷ��Լ��ݶ�����

		Eigen::DiagonalMatrix<float, 8> wM;		// ����Hessian����������׼����Ȩ�ؾ���

		Vec10f* JbBuffer;			// ��������������ʱ������0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
		Vec10f* JbBuffer_new;		// ��������������ʱ������0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.	

		Accumulator9 acc9;			// �����Ȳв�������λ���Լ���Ȳ������ֵ�Hessian
		Accumulator9 acc9SC;		// �����Ȳв������������Ȳ��ֵ�Hessian�������

		Vec3f dGrads[PYR_LEVELS];

		// ������ز���
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