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

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <algorithm>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include "IOWrapper/ImageDisplay.h"

#include "FullSystem/ResidualProjections.h"
#include "FullSystem/FullSystem.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{
	/// <summary>
	/// �Գ�Ա����activeResiduals�еĹ۲�в�������Ի�
	/// </summary>
	/// <param name="fixLinearization">�Ƿ����Ի�������ݵ���˹۲�в�ṹ��</param>
	/// <param name="toRemove">��Ա����activeResiduals���Ի�������õĲ���</param>
	/// <param name="min">�۲�в���������ֵ</param>
	/// <param name="max">�۲�в���������ֵ</param>
	/// <param name="stats">���߳�����������ֵ</param>
	/// <param name="tid">�߳�ID</param>
	void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
	{
		for (int k = min; k < max; ++k)
		{
			auto r = activeResiduals[k];
			if (r->stereoResidualFlag)
				(*stats)[0] += r->linearizeStereo(&Hcalib);
			else
				(*stats)[0] += r->linearize(&Hcalib);

			if (fixLinearization)
			{
				// �����в�۲�����Ի����ݲ����²в�״̬
				r->applyRes(true);

				// ���۲���ͼ��Χ�ڲ���Ȳв�����ֵ�����ڣ�����¸òв�Ķ�Ӧ�����������߳���
				// ���۲���ͼ��Χ����Ȳв�����ֵ�����⣬����Ϊ����۲�в������Ҫȥ��
				if (r->efResidual->isActive())
				{
					if (r->isNew && !r->stereoResidualFlag)
					{
						// �����������Ӳ�
						PointHessian* p = r->point;
						Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1);	// ����������ƽ��ʱ�����ص�����
						Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled;	// ��������ƽ����ͶӰʱ�����ص�����
						float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.

						// ������ͬ������˵���Ӳ�Խ�����Խ�󣬻���Խ����������ԽС
						if (relBS > p->maxRelBaseline)
							p->maxRelBaseline = relBS;

						p->numGoodResiduals++;
					}
				}
				else
				{
					toRemove[tid].emplace_back(activeResiduals[k]);
				}
			}
		}
	}

	/// <summary>
	/// �в����Ի�֮�����øòв������״̬������copyJacobians��ֵ�����ſɱ���Ϣ����˲в�ṹ��
	/// </summary>
	/// <param name="copyJacobians">�Ƿ�ǰ�˼�����ſɱ���Ϣ��������˲в�ṹ��</param>
	/// <param name="min">�в���������ֵ</param>
	/// <param name="max">�в���������ֵ</param>
	/// <param name="stats">���߳�����������ֵ</param>
	/// <param name="tid">�߳�ID</param>
	void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
	{
		for (int k = min; k < max; ++k)
			activeResiduals[k]->applyRes(true);
	}

	/// <summary>
	/// �趨����֡�Ĺ����ֵΪ����֡�۲⵽�����в����еĹ�Ȳв�ֵ
	/// </summary>
	void FullSystem::setNewFrameEnergyTH()
	{
		// collect all residuals and make decision on TH.
		allResVec.clear();
		allResVec.reserve(activeResiduals.size() * 2);
		FrameHessian* newFrame = frameHessians.back();

		for (PointFrameResidual* r : activeResiduals)
		{
			if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
				allResVec.emplace_back(r->state_NewEnergyWithOutlier);
		}

		// ����֡��û�й۲⵽֮ǰ�ؼ�֡�۲⵽����������ô����֡����ֵ��ֵ��ΪĬ��ֵ
		if (allResVec.empty())
		{
			newFrame->frameEnergyTH = 12 * 12 * patternNum;
			return;
		}

		int nthIdx = setting_frameEnergyTHN * allResVec.size();

		assert(nthIdx < (int)allResVec.size());
		assert(setting_frameEnergyTHN < 1);

		std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
		float nthElement = sqrtf(allResVec[nthIdx]);

		newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
		newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
		newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
		newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
	}

	/// <summary>
	/// ���Ի��в�����activeResiduals�����вв����������֡�й۲⵽������������
	/// ��������֡��Ȳв���ֵ��ɾ��������ϴ�Ĳв�
	/// </summary>
	/// <param name="fixLinearization"></param>
	/// <returns>activeResiduals�в����������вв�Ĺ�����֮��</returns>
	Vec3 FullSystem::linearizeAll(bool fixLinearization)
	{
		double lastEnergyP = 0;
		double lastEnergyR = 0;
		double num = 0;

		// 1���Թ۲�в���н������Ի�����ſɱ���Ϣ�����޳��۲�в�ϴ��۲�Խ��Ĳв�
		std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
		for (int it = 0; it < NUM_THREADS; ++it)
			toRemove[it].clear();

		if (setting_multiThreading)
		{
			treadReduce.reduce(std::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, activeResiduals.size(), 0);
			lastEnergyP = treadReduce.stats[0];
		}
		else
		{
			Vec10 stats;
			linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
			lastEnergyP = stats[0];
		}

		// 2�����û���������֡�Ĺ��������ֵ
		setNewFrameEnergyTH();

		// 3��������������۲�в��״̬����ɾ�в������й�����ϴ�Ĳв�
		if (fixLinearization)
		{
			// �����������²в�ʹ˴��²в�Ĺ۲�״̬
			for (PointFrameResidual* r : activeResiduals)
			{
				PointHessian* ph = r->point;
				if (ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].second = r->state_state;
				else if (ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].second = r->state_state;
			}

			// ɾ�����Ի��м�������Ĺ�Ȳв�����۲�Խ��Ĳв�
			int nResRemoved = 0;
			for (int it = 0; it < NUM_THREADS; ++it)
			{
				for (PointFrameResidual* r : toRemove[it])
				{
					PointHessian* ph = r->point;

					if (ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first = nullptr;
					else if (ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first = nullptr;

					for (unsigned int k = 0; k < ph->residuals.size(); ++k)
					{
						if (ph->residuals[k] == r)
						{
							ef->dropResidual(r->efResidual);
							deleteOut<PointFrameResidual>(ph->residuals, k);
							nResRemoved++;
							break;
						}
					}
				}
			}
		}

		return Vec3(lastEnergyP, lastEnergyR, num);
	}

	/// <summary>
	/// ������״̬��������ӵ���һ�������еõ���״̬���ϣ�������������С�ж��Ƿ���Ҫ��������
	/// </summary>
	/// <param name="stepfacC">����ڲ�������������</param>
	/// <param name="stepfacT">���λ��������������</param>
	/// <param name="stepfacR">�����̬������������</param>
	/// <param name="stepfacA">������������������</param>
	/// <param name="stepfacD">���������������������</param>
	/// <returns>�Ƿ��ܹ���ֹ����</returns>
	bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD)
	{
		Vec10 pstepfac;
		pstepfac.segment<3>(0).setConstant(stepfacT);
		pstepfac.segment<3>(3).setConstant(stepfacR);
		pstepfac.segment<4>(6).setConstant(stepfacA);

		float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0, sumNID = 0;

		if (setting_solverMode & SOLVER_MOMENTUM)
		{
			Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
			for (FrameHessian* fh : frameHessians)
			{
				Vec10 step = fh->step;
				step.head<6>() += 0.5f * (fh->step_backup.head<6>());

				fh->setState(fh->state_backup + step);
				sumA += step[6] * step[6];
				sumB += step[7] * step[7];
				sumT += step.segment<3>(0).squaredNorm();
				sumR += step.segment<3>(3).squaredNorm();

				for (PointHessian* ph : fh->pointHessians)
				{
					float step = ph->step + 0.5f * (ph->step_backup);
					ph->setIdepth(ph->idepth_backup + step);
					sumID += step * step;
					sumNID += fabsf(ph->idepth_backup);
					numID++;

					ph->setIdepthZero(ph->idepth_backup + step);
				}
			}
		}
		else if (setting_useOptimize)
		{
			Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
			T_WD_change = Sim3::exp(Vec7::Zero());

			if (setting_useImu)
			{
				state_twd += stepfacC * step_twd;
				if (std::exp(state_twd[6]) < 0.1 || std::exp(state_twd[6]) > 10)
				{
					initFailed = true;
					first_track_flag = false;
					return false;
				}

				T_WD_change = Sim3::exp(state_twd);

				/*Sim3 T_WD_temp = T_WD * T_WD_change;
				double s_temp = T_WD_temp.scale();
				double s_wd = T_WD.scale();
				double s_new = s_temp / s_wd;
				if (s_new > d_min)s_new = d_min;
				if (s_new < 1 / d_min)s_new = 1 / d_min;
				T_WD = Sim3(RxSO3(s_new * s_wd, T_WD_temp.rotationMatrix()), Vec3::Zero());*/

				T_WD = T_WD_l * T_WD_change;

				if (marg_num_half == 0)
				{
					T_WD_l = T_WD;
					state_twd.setZero();
				}
			}

			for (FrameHessian* fh : frameHessians)
			{
				fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
				sumA += fh->step[6] * fh->step[6];				// ��Ȳ���a����ƽ����
				sumB += fh->step[7] * fh->step[7];				// ��Ȳ���b����ƽ����
				sumT += fh->step.segment<3>(0).squaredNorm();	// �ؼ�֡λ�ò�������ƽ����
				sumR += fh->step.segment<3>(3).squaredNorm();	// �ؼ�֡��̬��������ƽ����

				if (setting_useImu)
				{
					fh->velocity += stepfacC * fh->step_imu.block(0, 0, 3, 1);
					fh->delta_bias_g += stepfacC * fh->step_imu.block(3, 0, 3, 1);
					fh->delta_bias_a += stepfacC * fh->step_imu.block(6, 0, 3, 1);
					fh->shell->velocity = fh->velocity;
					fh->shell->delta_bias_g = fh->delta_bias_g;
					fh->shell->delta_bias_a = fh->delta_bias_a;
				}

				for (PointHessian* ph : fh->pointHessians)
				{
					ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
					ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);

					sumID += ph->step * ph->step;				// ��������Ȳ�������ƽ����
					sumNID += fabsf(ph->idepth_backup);			// ���������֮��
					numID++;									// �����ؼ�֡������������
				}
			}
		}

		sumA /= frameHessians.size();
		sumB /= frameHessians.size();
		sumR /= frameHessians.size();
		sumT /= frameHessians.size();
		sumID /= numID;
		sumNID /= numID;

		if (!setting_debugout_runquiet)
			printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
				sqrtf(sumA) / (0.0005 * setting_thOptIterations),
				sqrtf(sumB) / (0.00005 * setting_thOptIterations),
				sqrtf(sumR) / (0.00005 * setting_thOptIterations),
				sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));

		EFDeltaValid = false;
		setPrecalcValues();

		// �������С����ֵ��˵��������ǰ��ֹ����������
		return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
			sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
			sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
			sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
	}

	/// <summary>
	/// ������һ�����������еõ��ĸ�������״̬����������ڲΡ����λ�ˡ���Ȳ����Լ����������
	/// </summary>
	/// <param name="backupLastStep">�Ƿ񱸷ݲ����ı�־λ</param>
	void FullSystem::backupState(bool backupLastStep)
	{
		if (setting_solverMode & SOLVER_MOMENTUM)
		{
			if (backupLastStep)
			{
				Hcalib.step_backup = Hcalib.step;
				Hcalib.value_backup = Hcalib.value;
				for (FrameHessian* fh : frameHessians)
				{
					fh->step_backup = fh->step;
					fh->state_backup = fh->get_state();
					for (PointHessian* ph : fh->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = ph->step;
					}
				}
			}
			else
			{
				Hcalib.step_backup.setZero();
				Hcalib.value_backup = Hcalib.value;
				for (FrameHessian* fh : frameHessians)
				{
					fh->step_backup.setZero();
					fh->state_backup = fh->get_state();
					for (PointHessian* ph : fh->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = 0;
					}
				}
			}
		}
		else
		{
			Hcalib.value_backup = Hcalib.value;
			for (FrameHessian* fh : frameHessians)
			{
				fh->state_backup = fh->get_state();
				for (PointHessian* ph : fh->pointHessians)
					ph->idepth_backup = ph->idepth;
			}
		}
	}

	/// <summary>
	/// ������ε������ɹ�����ôӦ�÷������ε�����������һ�ε������״̬
	/// </summary>
	void FullSystem::loadSateBackup()
	{
		Hcalib.setValue(Hcalib.value_backup);
		for (FrameHessian* fh : frameHessians)
		{
			fh->setState(fh->state_backup);
			for (PointHessian* ph : fh->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup);
				ph->setIdepthZero(ph->idepth_backup);
			}
		}

		EFDeltaValid = false;
		setPrecalcValues();
	}

	/// <summary>
	/// ��ӡ�����Ż���Ĳв���Ϣ
	/// </summary>
	/// <param name="res"></param>
	/// <param name="resL"></param>
	/// <param name="resM"></param>
	/// <param name="resPrior"></param>
	/// <param name="LExact"></param>
	/// <param name="a"></param>
	/// <param name="b"></param>
	void FullSystem::printOptRes(const Vec3& res, double resL, double resM, double resPrior, double LExact, float a, float b)
	{
		printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0], sqrtf((float)(res[0] / (patternNum * ef->resInA))), ef->resInA, ef->resInM, a, b);
	}

	float FullSystem::optimize(int numOptIts)
	{
		// 1�����ݻ����йؼ�֡������Լ�������Ż�����������
		// �������ؼ�ֻ֡��һ֡����ʱ�����Ż���
		// �������ؼ�֡���٣�˵��ϵͳ�ոճ�ʼ���ɹ���Ϊ�˱�֤���õĳ�ʼ���������������λ�ȡ���õĽ��
		// �������ؼ�֡�����Ѿ��Ƚ϶��ˣ�˵��ϵͳ�Ѿ����˳�ʼ���������������ˣ����������Ż�������������
		if (frameHessians.size() < 2) return 0;
		if (frameHessians.size() < 3) numOptIts = 20;
		if (frameHessians.size() < 4) numOptIts = 15;

		int numPoints = 0;
		int numLRes = 0;
		activeResiduals.clear();

		// 2��ϵͳ�н��в��Ϊ�����֣�����Ե��֡�ϵ�������Ӧ�Ĳв�ᱻ�̶����Ի��㣬�Ӷ�����һ�����Ի��㴦�Ĳв
		// �ڶ�������ͨ�в��Ե���в����ڼ����Ե����Ϣ�У�����ͨ�в����ڼ������Hessian��Ϣ
		for (FrameHessian* fh : frameHessians)
		{
			for (PointHessian* ph : fh->pointHessians)
			{
				for (PointFrameResidual* r : ph->residuals)
				{
					if (!r->efResidual->isLinearized)
					{
						activeResiduals.emplace_back(r);
						r->resetOOB();
					}
					else
						numLRes++;
				}
				numPoints++;
			}
		}

		if (!setting_debugout_runquiet)
			printf("OPTIMIZE %d pts, %d active res, %d linearize res!\n", ef->nPoints, (int)activeResiduals.size(), numLRes);

		// 3�����۲�в����Ի��õ��ſɱ���Ϣ������һ�λ����Ż���ʼ����ֵ
		Vec3 lastEnergy = linearizeAll(false);
		double lastEnergyL = calcLEnergy();		// ��Ե����Ϣ������ֵ
		double lastEnergyM = calcMEnergy();		// ���Ի��۲�в�����ֵ

		if (setting_multiThreading)
			treadReduce.reduce(std::bind(&FullSystem::applyRes_Reductor, this, true,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, activeResiduals.size(), 50);
		else
			applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

		if (!setting_debugout_runquiet)
		{
			printf("Initial Error\t");
			printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
		}

		//debugPlotTracking();

		double lambda = 1e-1;
		float stepsize = 1;
		VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
		for (int iteration = 0; iteration < numOptIts; ++iteration)
		{
			// ������һ�ε�������Ĳ�����������⵱ǰ���������������
			backupState(iteration != 0);
			solveSystem(iteration, lambda);

			// incDirChangeΪ���ε�����������֮��ļн�
			double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
			previousX = ef->lastX;

			if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
			{
				float newStepsize = exp(incDirChange * 1.4);
				if (incDirChange < 0 && stepsize > 1) stepsize = 1;

				stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
				if (stepsize > 2) stepsize = 2;
				if (stepsize < 0.25) stepsize = 0.25;
			}

			// ����ǰ������������������ӵ���Ӧ״̬�ϲ��жϲ��������Ƿ�С��������ֹ����
			bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

			// ���ݸ��º�ĸ���״̬�������Ի��۲�в����״̬���º�Ļ����Ż�����ֵ
			Vec3 newEnergy = linearizeAll(false);
			double newEnergyL = calcLEnergy();
			double newEnergyM = calcMEnergy();

			if (!setting_debugout_runquiet)
			{
				printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
					(newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
						lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
					iteration,
					log10(lambda),
					incDirChange,
					stepsize);
				printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
			}

			// �����ܱ��ε�������ô�ͼ�Сlambda�����������������ӿ��������
			// ���ܽӱ��ε�������ô�ͼӴ�lambda����С����������ʹ���������½�
			if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
			{
				if (setting_multiThreading)
					treadReduce.reduce(std::bind(&FullSystem::applyRes_Reductor, this, true,
						std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, activeResiduals.size(), 50);
				else
					applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

				lastEnergy = newEnergy;
				lastEnergyL = newEnergyL;
				lastEnergyM = newEnergyM;

				lambda *= 0.25;
			}
			else
			{
				loadSateBackup();
				lastEnergy = linearizeAll(false);
				lastEnergyL = calcLEnergy();
				lastEnergyM = calcMEnergy();
				lambda *= 1e2;
			}

			if (canbreak && iteration >= setting_minOptIterations) break;
		}

		// TODO
		Vec10 newStateZero = Vec10::Zero();
		newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);
		frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam, newStateZero);

		// 	frameHessians.back()->bias_a = frameHessians[frameHessians.size()-2]->bias_a + frameHessians[frameHessians.size()-2]->delta_bias_a;
		// 	frameHessians.back()->bias_g = frameHessians[frameHessians.size()-2]->bias_g + frameHessians[frameHessians.size()-2]->delta_bias_g;

		// 	Sim3 T_WD_temp = T_WD*T_WD_change;
		// 	double s_temp = T_WD_temp.scale();
		// 	double s_wd = T_WD.scale();
		// 	double s_new = s_temp/s_wd;
		// 	if(s_new>d_min)s_new = d_min;
		// 	if(s_new<1/d_min)s_new = 1/d_min;
		// 	T_WD = Sim3(RxSO3(s_new*s_wd,T_WD_temp.rotationMatrix()),Vec3::Zero());

		printf("INFO: T_WD.scale(): %f\n", T_WD.scale());
		printf("INFO: frameHessian.back()->bias_a: %f,%f,%f\n",
			(frameHessians.back()->bias_a + frameHessians.back()->delta_bias_a)[0],
			(frameHessians.back()->bias_a + frameHessians.back()->delta_bias_a)[1],
			(frameHessians.back()->bias_a + frameHessians.back()->delta_bias_a)[2]);
		printf("INFO: frameHessian.back()->bias_g: %f,%f,%f\n",
			(frameHessians.back()->bias_g + frameHessians.back()->delta_bias_g)[0],
			(frameHessians.back()->bias_g + frameHessians.back()->delta_bias_g)[1],
			(frameHessians.back()->bias_g + frameHessians.back()->delta_bias_g)[2]);

		EFDeltaValid = false;
		EFAdjointsValid = false;
		ef->setAdjointsF(&Hcalib);
		setPrecalcValues();

		lastEnergy = linearizeAll(true);

		if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
		{
			printf("KF Tracking failed: LOST!\n");
			isLost = true;
		}

		statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

		if (calibLog != 0)
		{
			(*calibLog) << Hcalib.value_scaled.transpose() <<
				" " << frameHessians.back()->get_state_scaled().transpose() <<
				" " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) <<
				" " << ef->resInM << "\n";
			calibLog->flush();
		}

		{
			std::unique_lock<std::mutex> crlock(shellPoseMutex);
			for (FrameHessian* fh : frameHessians)
			{
				fh->shell->camToWorld = fh->PRE_camToWorld;
				fh->shell->aff_g2l = fh->aff_g2l();
			}
		}

		//debugPlotTracking();

		// �����Ż����ƽ����ȹ۲�в�
		return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
	}

	/// <summary>
	/// �����Ż��еĵ��ε�������
	/// </summary>
	/// <param name="iteration">��ǰ��������</param>
	/// <param name="lambda">��ǰ��������������</param>
	void FullSystem::solveSystem(int iteration, double lambda)
	{
		ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

		ef->solveSystemF(iteration, lambda, &Hcalib);
	}

	/// <summary>
	/// �����Ե����Ϣ�Ż���Ȳв� 
	/// </summary>
	/// <returns>�Ż���������ֵ</returns>
	double FullSystem::calcMEnergy()
	{
		if (setting_forceAceptStep)
			return 0;

		return ef->calcMEnergyF();
	}

	/// <summary>
	/// �����Ż���������ֵ
	/// </summary>
	/// <returns>�Ż���������ֵ</returns>
	double FullSystem::calcLEnergy()
	{
		if (setting_forceAceptStep)
			return 0;

		return ef->calcLEnergyF_MT();
	}

	/// <summary>
	/// ɾ��ϵͳ�в��ܱ��۲⵽������
	/// </summary>
	void FullSystem::removeOutliers()
	{
		// 1����ǰ�˹���������н������۲⵽������ȥ������Ǹ�����ΪPS_DROP
		int numPointsDropped = 0;
		for (FrameHessian* fh : frameHessians)
		{
			for (unsigned int it = 0; it < fh->pointHessians.size(); ++it)
			{
				auto ph = fh->pointHessians[it];
				if (ph == nullptr) continue;

				if (ph->residuals.empty())
				{
					fh->pointHessiansOut.emplace_back(ph);
					fh->pointHessians[it] = fh->pointHessians.back();
					fh->pointHessians.pop_back();
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					it--;
					numPointsDropped++;
				}
			}
		}

		// 2���ں�˹���������н������ΪPS_DROP������ȥ��
		ef->dropPointsF();
	}

	/// <summary>
	/// ��ȡ�����ؼ�֡��������ռ�
	/// </summary>
	/// <param name="nullspaces_pose">λ�˲�����ռ�</param>
	/// <param name="nullspaces_scale">�߶Ȳ�����ռ�</param>
	/// <param name="nullspaces_affA">��Ȳ���a��ռ�</param>
	/// <param name="nullspaces_affB">��Ȳ���b��ռ�</param>
	/// <returns>ϵͳ��Ϣ������ռ����</returns>
	std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX>& nullspaces_pose,
		std::vector<VecX>& nullspaces_scale, std::vector<VecX>& nullspaces_affA, std::vector<VecX>& nullspaces_affB)
	{
		nullspaces_pose.clear();
		nullspaces_scale.clear();
		nullspaces_affA.clear();
		nullspaces_affB.clear();

		int n = CPARS + frameHessians.size() * 8;
		std::vector<VecX> nullspaces_x0_pre;

		// 1����ȡϵͳλ����ռ�
		for (int it = 0; it < 6; ++it)
		{
			VecX nullspace_x0(n);
			nullspace_x0.setZero();
			for (FrameHessian* fh : frameHessians)
			{
				nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(it);
				nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
				nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
			}
			nullspaces_x0_pre.emplace_back(nullspace_x0);
			nullspaces_pose.emplace_back(nullspace_x0);
		}

		// 2����ȡϵͳ��Ȳ�����ռ�
		for (int it = 0; it < 2; ++it)
		{
			VecX nullspace_x0(n);
			nullspace_x0.setZero();
			for (FrameHessian* fh : frameHessians)
			{
				nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(it).head<2>();
				nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
				nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
			}
			nullspaces_x0_pre.emplace_back(nullspace_x0);
			if (it == 0) nullspaces_affA.emplace_back(nullspace_x0);
			if (it == 1) nullspaces_affB.emplace_back(nullspace_x0);
		}

		// ��ȡϵͳ�߶���ռ�
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for (FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
			nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.emplace_back(nullspace_x0);
		nullspaces_scale.emplace_back(nullspace_x0);

		return nullspaces_x0_pre;
	}
}