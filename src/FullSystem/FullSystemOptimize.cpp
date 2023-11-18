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
	/// 对成员变量activeResiduals中的观测残差进行线性化
	/// </summary>
	/// <param name="fixLinearization">是否将线性化结果传递到后端观测残差结构中</param>
	/// <param name="toRemove">成员变量activeResiduals线性化结果不好的部分</param>
	/// <param name="min">观测残差序列索引值</param>
	/// <param name="max">观测残差序列索引值</param>
	/// <param name="stats">多线程任务结果返回值</param>
	/// <param name="tid">线程ID</param>
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
				// 拷贝残差观测的线性化数据并更新残差状态
				r->applyRes(true);

				// 若观测在图像范围内并光度残差在阈值条件内，则更新该残差的对应特征的最大基线长度
				// 若观测在图像范围外或光度残差在阈值条件外，则认为这个观测残差并不好需要去掉
				if (r->efResidual->isActive())
				{
					if (r->isNew && !r->stereoResidualFlag)
					{
						// 计算特征的视差
						PointHessian* p = r->point;
						Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1);	// 假设特征无平移时的像素点坐标
						Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled;	// 按照正常平移量投影时的像素点坐标
						float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.

						// 对于相同特征来说，视差越大基线越大，基线越大则深度误差越小
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
	/// 残差线性化之后设置该残差的最新状态并根据copyJacobians标值拷贝雅可比信息到后端残差结构中
	/// </summary>
	/// <param name="copyJacobians">是否将前端计算的雅可比信息拷贝到后端残差结构中</param>
	/// <param name="min">残差序列索引值</param>
	/// <param name="max">残差序列索引值</param>
	/// <param name="stats">多线程任务结果返回值</param>
	/// <param name="tid">线程ID</param>
	void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
	{
		for (int k = min; k < max; ++k)
			activeResiduals[k]->applyRes(true);
	}

	/// <summary>
	/// 设定最新帧的光度阈值为被该帧观测到特征残差序列的光度残差值
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

		// 最新帧中没有观测到之前关键帧观测到的特征，那么最新帧能量值阈值设为默认值
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
	/// 线性化残差序列activeResiduals中所有残差，根据在最新帧中观测到的特征光度误差
	/// 计算最新帧光度残差阈值并删除光度误差较大的残差
	/// </summary>
	/// <param name="fixLinearization"></param>
	/// <returns>activeResiduals残差序列中所有残差的光度误差之和</returns>
	Vec3 FullSystem::linearizeAll(bool fixLinearization)
	{
		double lastEnergyP = 0;
		double lastEnergyR = 0;
		double num = 0;

		// 1、对观测残差进行进行线性化求解雅可比信息，并剔除观测残差较大或观测越界的残差
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

		// 2、设置滑窗中最新帧的光度误差的阈值
		setNewFrameEnergyTH();

		// 3、更新特征最近观测残差的状态，并删残差序列中光度误差较大的残差
		if (fixLinearization)
		{
			// 更新特征最新残差和此次新残差的观测状态
			for (PointFrameResidual* r : activeResiduals)
			{
				PointHessian* ph = r->point;
				if (ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].second = r->state_state;
				else if (ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].second = r->state_state;
			}

			// 删除线性化中计算出来的光度残差过大或观测越界的残差
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
	/// 将各个状态的增量添加到上一步迭代中得到的状态量上，并根据增量大小判断是否还需要继续迭代
	/// </summary>
	/// <param name="stepfacC">相机内参增量控制因子</param>
	/// <param name="stepfacT">相机位置增量控制因子</param>
	/// <param name="stepfacR">相机姿态增量控制因子</param>
	/// <param name="stepfacA">相机光度增量控制因子</param>
	/// <param name="stepfacD">特征逆深度增量控制因子</param>
	/// <returns>是否能够中止迭代</returns>
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
				sumA += fh->step[6] * fh->step[6];				// 光度参数a增量平方和
				sumB += fh->step[7] * fh->step[7];				// 光度参数b增量平方和
				sumT += fh->step.segment<3>(0).squaredNorm();	// 关键帧位置参数增量平方和
				sumR += fh->step.segment<3>(3).squaredNorm();	// 关键帧姿态参数增量平方和

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

					sumID += ph->step * ph->step;				// 特征逆深度参数增量平方和
					sumNID += fabsf(ph->idepth_backup);			// 特征逆深度之和
					numID++;									// 滑窗关键帧中特征总数量
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

		// 如果增量小于阈值，说明可以提前中止迭代过程了
		return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
			sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
			sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
			sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
	}

	/// <summary>
	/// 备份上一个迭代步骤中得到的各个参数状态：包含相机内参、相机位姿、光度参数以及特征逆深度
	/// </summary>
	/// <param name="backupLastStep">是否备份参数的标志位</param>
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
	/// 如果本次迭代不成功，那么应该放弃本次迭代并加载上一次迭代后的状态
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
	/// 打印滑窗优化后的残差信息
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
		// 1、根据滑窗中关键帧的数量约定滑窗优化迭代次数：
		// 若滑窗关键帧只有一帧，此时不用优化；
		// 若滑窗关键帧较少，说明系统刚刚初始化成功，为了保证良好的初始化结果，多迭代几次获取更好的结果
		// 若滑窗关键帧数量已经比较多了，说明系统已经过了初始化环节正常跟踪了，正常设置优化迭代数量即可
		if (frameHessians.size() < 2) return 0;
		if (frameHessians.size() < 3) numOptIts = 20;
		if (frameHessians.size() < 4) numOptIts = 15;

		int numPoints = 0;
		int numLRes = 0;
		activeResiduals.clear();

		// 2、系统中将残差分为了两种：被边缘化帧上的特征对应的残差会被固定线性化点，从而计算一个线性化点处的残差；
		// 第二种是普通残差；边缘化残差用于计算边缘化信息中，而普通残差用于计算基本Hessian信息
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

		// 3、将观测残差线性化得到雅可比信息并计算一次滑窗优化初始能量值
		Vec3 lastEnergy = linearizeAll(false);
		double lastEnergyL = calcLEnergy();		// 边缘化信息的能量值
		double lastEnergyM = calcMEnergy();		// 线性化观测残差能量值

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
			// 备份上一次迭代步骤的参数增量并求解当前迭代步骤参数增量
			backupState(iteration != 0);
			solveSystem(iteration, lambda);

			// incDirChange为两次迭代求解的增量之间的夹角
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

			// 将当前迭代步骤参数增量附加到对应状态上并判断参数增量是否小到可以中止迭代
			bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

			// 根据更新后的各个状态重新线性化观测残差并计算状态更新后的滑窗优化能量值
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

			// 若接受本次迭代，那么就减小lambda，增大增量步长，加快迭代收敛
			// 若拒接本次迭代，那么就加大lambda，减小增量步长，使能量快速下降
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

		// 滑窗优化后的平均光度观测残差
		return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
	}

	/// <summary>
	/// 滑窗优化中的单次迭代过程
	/// </summary>
	/// <param name="iteration">当前迭代次数</param>
	/// <param name="lambda">当前迭代的阻尼因子</param>
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
	/// 计算边缘化信息优化光度残差 
	/// </summary>
	/// <returns>优化函数能量值</returns>
	double FullSystem::calcMEnergy()
	{
		if (setting_forceAceptStep)
			return 0;

		return ef->calcMEnergyF();
	}

	/// <summary>
	/// 计算优化函数能量值
	/// </summary>
	/// <returns>优化函数能量值</returns>
	double FullSystem::calcLEnergy()
	{
		if (setting_forceAceptStep)
			return 0;

		return ef->calcLEnergyF_MT();
	}

	/// <summary>
	/// 删除系统中不能被观测到的特征
	/// </summary>
	void FullSystem::removeOutliers()
	{
		// 1、在前端管理的特征中将不被观测到的特征去除并标记该特征为PS_DROP
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

		// 2、在后端管理的特征中将被标记为PS_DROP的特征去除
		ef->dropPointsF();
	}

	/// <summary>
	/// 获取滑窗关键帧参数的零空间
	/// </summary>
	/// <param name="nullspaces_pose">位姿参数零空间</param>
	/// <param name="nullspaces_scale">尺度参数零空间</param>
	/// <param name="nullspaces_affA">光度参数a零空间</param>
	/// <param name="nullspaces_affB">光度参数b零空间</param>
	/// <returns>系统信息矩阵零空间基底</returns>
	std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX>& nullspaces_pose,
		std::vector<VecX>& nullspaces_scale, std::vector<VecX>& nullspaces_affA, std::vector<VecX>& nullspaces_affB)
	{
		nullspaces_pose.clear();
		nullspaces_scale.clear();
		nullspaces_affA.clear();
		nullspaces_affB.clear();

		int n = CPARS + frameHessians.size() * 8;
		std::vector<VecX> nullspaces_x0_pre;

		// 1、获取系统位姿零空间
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

		// 2、获取系统光度参数零空间
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

		// 获取系统尺度零空间
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