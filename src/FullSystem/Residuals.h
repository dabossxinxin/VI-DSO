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

#include "vector"
#include <iostream>
#include <fstream>

#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
	class PointHessian;
	class FrameHessian;
	class CalibHessian;
	class PointFrameHessian;
	class EFResidual;

	enum ResLocation
	{
		ACTIVE = 0,
		LINEARIZED,
		MARGINALIZED,
		NONE
	};

	struct FullJacRowT
	{
		Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
	};

	// ��ͼ����ؼ�֮֡��Ĺ�Ȳв�
	class PointFrameResidual
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		EFResidual* efResidual;				// ��EFResidual������жԷ�ָ��ʵ��ͨ��			
		bool stereoResidualFlag = false;	// ��ǰ�в��Ƿ�Ϊ˫Ŀ�в�ı�־λ
		static int instanceCounter;			// �в����������ľ�̬��Ա����

		ResState state_state;				// TODO:��ǰ�в��״̬
		double state_energy;
		ResState state_NewState;
		double state_NewEnergy;
		double state_NewEnergyWithOutlier;

		PointHessian* point;				// ���쵱ǰ�в�ĵ�ͼ��
		FrameHessian* host;					// ������ǰ�в��ͼ�����֡
		FrameHessian* target;				// ������ǰ�в��ͼ���ͶӰ֡
		RawResidualJacobian* J;				// ��ǰ�в���������Ż��������ſɱ�

		void setState(ResState s) { state_state = s; }

		bool isNew;											// ��ǰ�۲�в��Ƿ����¹���ģ��ڹ��캯���н����ֵ��Ϊtrue

		Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];		// �����㼰��Χ������߸���target֡�ϵ�ͶӰ��������
		Vec3f centerProjectedTo;							// �������ĵ���target֡�ϵ�ͶӰ���ص�����[0,1]�Լ������[2]

		~PointFrameResidual();
		PointFrameResidual();
		PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);

		double linearize(CalibHessian* HCalib);
		double linearizeStereo(CalibHessian* HCalib);

		void resetOOB()
		{
			state_NewEnergy = state_energy = 0;
			state_NewState = ResState::OUTLIER;
			setState(ResState::INNER);
		};

		void applyRes(bool copyJacobians);
		void debugPlot();
	};
}