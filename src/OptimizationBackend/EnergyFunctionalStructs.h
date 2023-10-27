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

#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
	class EFFrame;
	class EFPoint;
	class EFResidual;
		
	class FrameHessian;
	class PointHessian;
	class CalibHessian;

	class PointFrameResidual;
	class EnergyFunctional;

	/// <summary>
	/// ����й���Ĺ۲�в����ݽṹ
	/// </summary>
	class EFResidual
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
			data(org), point(point_), host(host_), target(target_)
		{
			isLinearized = false;
			isActiveAndIsGoodNEW = false;
			J = new RawResidualJacobian();
			assert(((long)this) % 16 == 0);
			assert(((long)J) % 16 == 0);
		}
		inline ~EFResidual()
		{
			delete J;
		}

		void takeDataF();

		void fixLinearizationF(EnergyFunctional* ef);

		// structural pointers
		PointFrameResidual* data;		// ��ǰ�в��Ӧ�Ĺ۲�������ǰ�˸����еı�ʾ
		int hostIDX;					// ��ǰ�в��Ӧ�۲���������֡�ڻ����ؼ�֡�����е�ID
		int targetIDX;					// ��ǰ�в��Ӧ�۲�������Ŀ��֡�ڻ����ؼ�֡�����е�ID
		EFPoint* point;					// ��ǰ�в��Ӧ�Ĺ۲������ڻ����Ż��еı�ʾ
		EFFrame* host;					// ��ǰ�в��Ӧ�۲�������֡�ڻ����Ż��еı�ʾ
		EFFrame* target;				// ��ǰ�в��Ӧ�۲�����Ŀ��֡�ڻ����Ż��еı�ʾ
		int idxInAll;					// �۲�в��������в������е����

		RawResidualJacobian* J;			// ��ǰ�۲�в���ڲΡ�λ�ˡ���Ȳ����Լ�����ȵ��ſɱ�

		// 	VecNRf res_toZeroF;
		// 	Vec8f JpJdF=Vec8f::Zero();
		EIGEN_ALIGN16 VecNRf res_toZeroF;
		EIGEN_ALIGN16 Vec8f JpJdF = Vec8f::Zero();		// p��ʾ֡λ���Լ���Ȳ�����d��ʾ���������

		// status.
		bool isLinearized;

		// if residual is not OOB & not OUTLIER & should be used during accumulations
		bool isActiveAndIsGoodNEW;
		inline const bool &isActive() const { return isActiveAndIsGoodNEW; }
	};

	enum EFPointStatus { PS_GOOD = 0, PS_MARGINALIZE, PS_DROP };

	class EFPoint
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFPoint(PointHessian* d, EFFrame* host_) : data(d), host(host_)
		{
			takeData();
			stateFlag = EFPointStatus::PS_GOOD;
		}
		void takeData();

		PointHessian* data;				// ������ǰ���еı�ʾ��ʽ

		float priorF;					// ����������Ϣ
		float deltaF;					// �����ڻ����Ż��е��Ż�����
			
		int idxInPoints;				// �����ڶ�Ӧ��֡��������������е����
		EFFrame* host;					// ������Ӧ����֡

		std::vector<EFResidual*> residualsAll;	// ��������Ĺ۲�в�����

		float bdSumF;
		float HdiF = 1e-3;		// TODO
		float Hdd_accLF;		// �������Ի�״̬�µ������Hessian
		VecCf Hcd_accLF;		// �������Ի�״̬�µ������&����ڲ�Hessian
		float bd_accLF;			// �������Ի�״̬�µ������b
		float Hdd_accAF;		// ��������״̬�µ������Hessian
		VecCf Hcd_accAF;		// ��������״̬�µ������&����ڲ�Hessian
		float bd_accAF;			// ��������״̬�µ������b

		EFPointStatus stateFlag;
	};

	/// <summary>
	/// ����й���Ĺؼ�֡������ʽ
	/// </summary>
	class EFFrame
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFFrame(FrameHessian* d) : data(d)
		{
			takeData();
		}
		void takeData();

		Vec8 prior;				// prior hessian (diagonal)
		Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
		Vec8 delta;				// state - state_zero.

		std::vector<EFPoint*> points;	// �ؼ�֡�������������
		FrameHessian* data;				// �ؼ�֡��ǰ���еı�ʾ

		int idx;						// �ؼ�֡�ڻ����ؼ�֡�е����
		int frameID;					// �ؼ�֡��ȫ���е�֡���
		bool m_flag = false;
	};
}