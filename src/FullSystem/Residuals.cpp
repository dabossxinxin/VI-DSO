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

#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include "FullSystem/FullSystem.h"
//#include "FullSystem/Residuals.h"
//#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ResidualProjections.h"

//#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace dso
{
	int PointFrameResidual::instanceCounter = 0;

	PointFrameResidual::PointFrameResidual()
	{
		assert(false);
		instanceCounter++;
	}

	PointFrameResidual::~PointFrameResidual()
	{
		assert(efResidual == 0);
		instanceCounter--;
		delete J;
	}

	PointFrameResidual::PointFrameResidual(PointHessian* point, FrameHessian* host, FrameHessian* target)
		: point(point), host(host), target(target)
	{
		efResidual = 0;
		instanceCounter++;
		resetOOB();
		J = new RawResidualJacobian();
		assert(((long)J) % 16 == 0);

		isNew = true;
	}

	/// <summary>
	/// 线性化当前残差并判断当前观测残差的状态：即求解当前观测残差相对于各个参数的雅可比；
	/// ResState::OOB：当观测残差的老状态为OOB时或特征在target帧中的投影越界时标记观测状态为OOB；
	/// ResState::OUTLIER：当观测残差能量值高于host或target帧能量阈值或在target帧中投影像素点梯度过低时标记为OUTLIER；
	/// ResState::INNER：当观测残差能量值低于host和target帧能量阈值且在target帧中投影像素点梯度不低于阈值标记为INNER；
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	/// <returns>当前观测残差能量值</returns>
	double PointFrameResidual::linearize(CalibHessian* HCalib)
	{
		state_NewEnergyWithOutlier = -1;

		// 1、若观测残差状态标记为超出图像，那么这种残差不参与线性化
		if (state_state == ResState::OOB)
		{
			state_NewState = ResState::OOB;
			return state_energy;
		}

		float energyLeft = 0;
		FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
		const Eigen::Vector3f* dIl = target->dI;
		const float * const color = point->color;
		const float * const weights = point->weights;

		const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
		const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
		const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
		const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
		
		Vec2f affLL = precalc->PRE_aff_mode;
		float b0 = precalc->PRE_b0_mode;

		// 1、计算特征中心点像素坐标相对于位姿、相机内参的雅可比
		Vec6f d_xi_x, d_xi_y;	// 特征在target上的投影像素点对hostToTarget位姿的雅可比
		Vec4f d_C_x, d_C_y;		// 特征在target上的投影像素点对相机内参的雅可比
		float d_d_x, d_d_y;		// 特征在target上的投影像素点对该特征在host帧逆深度的雅可比
		{
			float drescale, u, v, new_idepth;
			float Ku, Kv;
			Vec3f KliP;

			// TODO 使用线性化点处的位姿投影特征点
			if (!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0, HCalib,
				PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{
				state_NewState = ResState::OOB;
				return state_energy;
			}

			centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

			d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u)*SCALE_IDEPTH*HCalib->fxl();
			d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v)*SCALE_IDEPTH*HCalib->fyl();

			d_C_x[2] = drescale * (PRE_RTll_0(2, 0)*u - PRE_RTll_0(0, 0));
			d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2, 1)*u - PRE_RTll_0(0, 1)) * HCalib->fyli();
			d_C_x[0] = KliP[0] * d_C_x[2];
			d_C_x[1] = KliP[1] * d_C_x[3];

			d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2, 0)*v - PRE_RTll_0(1, 0)) * HCalib->fxli();
			d_C_y[3] = drescale * (PRE_RTll_0(2, 1)*v - PRE_RTll_0(1, 1));
			d_C_y[0] = KliP[0] * d_C_y[2];
			d_C_y[1] = KliP[1] * d_C_y[3];

			d_C_x[0] = (d_C_x[0] + u)*SCALE_F;
			d_C_x[1] *= SCALE_F;
			d_C_x[2] = (d_C_x[2] + 1)*SCALE_C;
			d_C_x[3] *= SCALE_C;

			d_C_y[0] *= SCALE_F;
			d_C_y[1] = (d_C_y[1] + v)*SCALE_F;
			d_C_y[2] *= SCALE_C;
			d_C_y[3] = (d_C_y[3] + 1)*SCALE_C;

			d_xi_x[0] = new_idepth * HCalib->fxl();
			d_xi_x[1] = 0;
			d_xi_x[2] = -new_idepth * u*HCalib->fxl();
			d_xi_x[3] = -u * v*HCalib->fxl();
			d_xi_x[4] = (1 + u * u)*HCalib->fxl();
			d_xi_x[5] = -v * HCalib->fxl();

			d_xi_y[0] = 0;
			d_xi_y[1] = new_idepth * HCalib->fyl();
			d_xi_y[2] = -new_idepth * v*HCalib->fyl();
			d_xi_y[3] = -(1 + v * v)*HCalib->fyl();
			d_xi_y[4] = u * v*HCalib->fyl();
			d_xi_y[5] = u * HCalib->fyl();
		}

		{
			J->Jpdxi[0] = d_xi_x;
			J->Jpdxi[1] = d_xi_y;

			J->Jpdc[0] = d_C_x;
			J->Jpdc[1] = d_C_y;

			J->Jpdd[0] = d_d_x;
			J->Jpdd[1] = d_d_y;
		}

		float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
		float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
		float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;
		float wJI2_sum = 0;

		// 2、计算特征点领域特征光度残差相对于target上像素坐标以及hostToTarget光度参数的导数
		for (int idx = 0; idx < patternNum; ++idx)
		{
			float Ku, Kv;

			// TODO 使用实际位姿投影特征
			if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1], 
				point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{
				state_NewState = ResState::OOB; 
				return state_energy;
			}

			projectedTo[idx][0] = Ku;
			projectedTo[idx][1] = Kv;

			Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
			float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);

			// 残差相对于光度参数a21的导数
			float drdA = (color[idx] - b0);
			if (!std::isfinite((float)hitColor[0]))
			{
				state_NewState = ResState::OOB; 
				return state_energy;
			}

			float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
			w = 0.5f*(w + weights[idx]);

			float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
			energyLeft += w * w*hw *residual*residual*(2 - hw);

			{
				if (hw < 1) hw = sqrtf(hw);
				hw = hw * w;

				hitColor[1] *= hw;
				hitColor[2] *= hw;

				J->resF[idx] = residual * hw;

				J->JIdx[0][idx] = hitColor[1];
				J->JIdx[1][idx] = hitColor[2];
				J->JabF[0][idx] = drdA * hw;
				J->JabF[1][idx] = hw;

				JIdxJIdx_00 += hitColor[1] * hitColor[1];
				JIdxJIdx_11 += hitColor[2] * hitColor[2];
				JIdxJIdx_10 += hitColor[1] * hitColor[2];

				JabJIdx_00 += drdA * hw * hitColor[1];
				JabJIdx_01 += drdA * hw * hitColor[2];
				JabJIdx_10 += hw * hitColor[1];
				JabJIdx_11 += hw * hitColor[2];

				JabJab_00 += drdA * drdA*hw*hw;
				JabJab_01 += drdA * hw*hw;
				JabJab_11 += hw * hw;

				wJI2_sum += hw * hw*(hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

				// 如果不优化光度参数a和b，那么直接设置对应的雅可比为0
				if (setting_affineOptModeA < 0) J->JabF[0][idx] = 0;
				if (setting_affineOptModeB < 0) J->JabF[1][idx] = 0;
			}
		}

		J->JIdx2(0, 0) = JIdxJIdx_00;
		J->JIdx2(0, 1) = JIdxJIdx_10;
		J->JIdx2(1, 0) = JIdxJIdx_10;
		J->JIdx2(1, 1) = JIdxJIdx_11;
		J->JabJIdx(0, 0) = JabJIdx_00;
		J->JabJIdx(0, 1) = JabJIdx_01;
		J->JabJIdx(1, 0) = JabJIdx_10;
		J->JabJIdx(1, 1) = JabJIdx_11;
		J->Jab2(0, 0) = JabJab_00;
		J->Jab2(0, 1) = JabJab_01;
		J->Jab2(1, 0) = JabJab_01;
		J->Jab2(1, 1) = JabJab_11;

		state_NewEnergyWithOutlier = energyLeft;

		// 通过能量值判断当前观测残差是否为内点
		if (energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
		{
			energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
			state_NewState = ResState::OUTLIER;
		}
		else
		{
			state_NewState = ResState::INNER;
		}

		state_NewEnergy = energyLeft;
		return energyLeft;
	}

	double PointFrameResidual::linearizeStereo(CalibHessian* HCalib)
	{
		state_NewEnergyWithOutlier = -1;

		if (state_state == ResState::OOB)
		{
			state_NewState = ResState::OOB; return state_energy;
		}

		// 	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
		float energyLeft = 0;
		const Eigen::Vector3f* dIl = target->dI;
		//const float* const Il = target->I;

		// 	Mat33f PRE_RTll_0 = Mat33f::Identity();
		// 	Vec3f PRE_tTll_0 = Vec3f::Zero();
		// 	PRE_tTll_0 << -baseline, 0, 0;
		// 	Mat33f PRE_RTll = Mat33f::Identity();
		// 	Vec3f PRE_tTll = PRE_tTll_0;
		// 	Mat33f K = Mat33f::Zero();
		// 	K(0,0) = HCalib->fxl();
		// 	K(1,1) = HCalib->fyl();
		// 	K(0,2) = HCalib->cxl();
		// 	K(1,2) = HCalib->cyl();
		// 	K(2,2) = 1;
		// 	Mat33f PRE_KRKiTll = K * PRE_RTll * K.inverse();
		// 	Mat33f PRE_RKiTll = PRE_RTll * K.inverse();
		// 	Vec3f PRE_KtTll = K * PRE_tTll;
		// 	Mat44 M_C1C0_2 = T_WD.inverse().matrix()*T_C1C0.matrix()*T_WD.matrix();
		// 	SE3 T_C1C02(M_C1C0_2);
		Mat33f PRE_RTll_0 = T_C1C0.rotationMatrix().cast<float>();
		Vec3f PRE_tTll_0 = T_C1C0.translation().cast<float>();

		Mat33f PRE_RTll = T_C1C0.rotationMatrix().cast<float>();
		// 	Mat33f PRE_RTll_0 = T_C1C02.rotationMatrix().cast<float>();
		// 	Vec3f PRE_tTll_0 = T_C1C02.translation().cast<float>();
		// 
		// 	Mat33f PRE_RTll = T_C1C02.rotationMatrix().cast<float>();
		Vec3f PRE_tTll = PRE_tTll_0;
		Mat33f K = Mat33f::Zero();
		K(0, 0) = HCalib->fxl();
		K(1, 1) = HCalib->fyl();
		K(0, 2) = HCalib->cxl();
		K(1, 2) = HCalib->cyl();
		K(2, 2) = 1;
		Mat33f PRE_KRKiTll = K_right * PRE_RTll * K.inverse();
		Mat33f PRE_RKiTll = PRE_RTll * K.inverse();
		Vec3f PRE_KtTll = K_right * PRE_tTll;
		// 	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
		// 	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
		// 	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
		// 	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
		const float * const color = point->color;
		const float * const weights = point->weights;

		// 	Vec2f affLL = precalc->PRE_aff_mode;
		// 	float b0 = precalc->PRE_b0_mode;
		Vec2f affLL;
		affLL << 1, 0;
		float b0 = 0;

		Vec6f d_xi_x, d_xi_y;
		Vec4f d_C_x, d_C_y;
		float d_d_x, d_d_y;
		{
			float drescale, u, v, new_idepth;
			float Ku, Kv;
			Vec3f KliP;

			if (!projectPoint(point->u, point->v, point->idepth_zero_scaled / T_WD.scale(), 0, 0, HCalib,
				PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{
				state_NewState = ResState::OOB; return state_energy;
			}

			centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

			// diff d_idepth
			d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u)*SCALE_IDEPTH*HCalib->fxl();
			d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v)*SCALE_IDEPTH*HCalib->fyl();

			// diff calib
			d_C_x[2] = drescale * (PRE_RTll_0(2, 0)*u - PRE_RTll_0(0, 0));
			d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2, 1)*u - PRE_RTll_0(0, 1)) * HCalib->fyli();
			d_C_x[0] = KliP[0] * d_C_x[2];
			d_C_x[1] = KliP[1] * d_C_x[3];

			d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2, 0)*v - PRE_RTll_0(1, 0)) * HCalib->fxli();
			d_C_y[3] = drescale * (PRE_RTll_0(2, 1)*v - PRE_RTll_0(1, 1));
			d_C_y[0] = KliP[0] * d_C_y[2];
			d_C_y[1] = KliP[1] * d_C_y[3];

			d_C_x[0] = (d_C_x[0] + u)*SCALE_F;
			d_C_x[1] *= SCALE_F;
			d_C_x[2] = (d_C_x[2] + 1)*SCALE_C;
			d_C_x[3] *= SCALE_C;

			d_C_y[0] *= SCALE_F;
			d_C_y[1] = (d_C_y[1] + v)*SCALE_F;
			d_C_y[2] *= SCALE_C;
			d_C_y[3] = (d_C_y[3] + 1)*SCALE_C;

			d_xi_x[0] = new_idepth * HCalib->fxl();
			d_xi_x[1] = 0;
			d_xi_x[2] = -new_idepth * u*HCalib->fxl();
			d_xi_x[3] = -u * v*HCalib->fxl();
			d_xi_x[4] = (1 + u * u)*HCalib->fxl();
			d_xi_x[5] = -v * HCalib->fxl();

			d_xi_y[0] = 0;
			d_xi_y[1] = new_idepth * HCalib->fyl();
			d_xi_y[2] = -new_idepth * v*HCalib->fyl();
			d_xi_y[3] = -(1 + v * v)*HCalib->fyl();
			d_xi_y[4] = u * v*HCalib->fyl();
			d_xi_y[5] = u * HCalib->fyl();
		}

		{
			J->Jpdxi[0] = Vec6f::Zero();
			J->Jpdxi[1] = Vec6f::Zero();

			J->Jpdc[0] = Vec4f::Zero();
			J->Jpdc[1] = Vec4f::Zero();

			J->Jpdd[0] = d_d_x / T_WD.scale();
			J->Jpdd[1] = d_d_y / T_WD.scale();
		}

		float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
		float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
		float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

		float wJI2_sum = 0;
		float lambda = setting_stereoWeight;
		if (!setting_useStereo) lambda = 1.0;

		for (int idx = 0; idx < patternNum; idx++)
		{
			float Ku, Kv;
			if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1], point->idepth_scaled / T_WD.scale(), PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{
				state_NewState = ResState::OOB; return state_energy;
			}

			projectedTo[idx][0] = Ku;
			projectedTo[idx][1] = Kv;

			Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
			float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);

			float drdA = (color[idx] - b0);
			if (!std::isfinite((float)hitColor[0]))
			{
				state_NewState = ResState::OOB;
				return state_energy;
			}

			float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
			w = 0.5f*(w + weights[idx]);

			float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
			hw = hw * lambda;
			energyLeft += w * w*hw *residual*residual*(2 - hw);

			{
				if (hw < 1) hw = sqrtf(hw);
				hw = hw * w;

				hitColor[1] *= hw;
				hitColor[2] *= hw;

				J->resF[idx] = residual * hw;

				J->JIdx[0][idx] = hitColor[1];
				J->JIdx[1][idx] = hitColor[2];
				J->JabF[0][idx] = 0;
				J->JabF[1][idx] = 0;

				JIdxJIdx_00 += hitColor[1] * hitColor[1];
				JIdxJIdx_11 += hitColor[2] * hitColor[2];
				JIdxJIdx_10 += hitColor[1] * hitColor[2];

				JabJIdx_00 += drdA * hw * hitColor[1];
				JabJIdx_01 += drdA * hw * hitColor[2];
				JabJIdx_10 += hw * hitColor[1];
				JabJIdx_11 += hw * hitColor[2];

				JabJab_00 += drdA * drdA*hw*hw;
				JabJab_01 += drdA * hw*hw;
				JabJab_11 += hw * hw;

				wJI2_sum += hw * hw*(hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

				if (setting_affineOptModeA < 0) J->JabF[0][idx] = 0;
				if (setting_affineOptModeB < 0) J->JabF[1][idx] = 0;
			}
		}

		J->JIdx2(0, 0) = JIdxJIdx_00;
		J->JIdx2(0, 1) = JIdxJIdx_10;
		J->JIdx2(1, 0) = JIdxJIdx_10;
		J->JIdx2(1, 1) = JIdxJIdx_11;
		J->JabJIdx(0, 0) = 0;
		J->JabJIdx(0, 1) = 0;
		J->JabJIdx(1, 0) = 0;
		J->JabJIdx(1, 1) = 0;
		J->Jab2(0, 0) = 0;
		J->Jab2(0, 1) = 0;
		J->Jab2(1, 0) = 0;
		J->Jab2(1, 1) = 0;

		state_NewEnergyWithOutlier = energyLeft;

		if (energyLeft > lambda*std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < lambda * 2)
		{
			energyLeft = lambda * std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
			state_NewState = ResState::OUTLIER;
		}
		else
		{
			state_NewState = ResState::INNER;
		}

		state_NewEnergy = energyLeft;
		return energyLeft;
	}

	/// <summary>
	/// 在观测残差的target帧中绘制观测到的特征在target帧中的投影
	/// 若在图上特征颜色为蓝色，表示当前残差状态为INNER
	/// 若在图上特征颜色为红色，表示当前残差状态为OUTLIER
	/// </summary>
	void PointFrameResidual::debugPlot()
	{
		if (state_state == ResState::OOB) return;
		Vec3b cT = Vec3b(0, 0, 0);

		if (freeDebugParam5 == 0)
		{
			float rT = 20 * sqrt(state_energy / 9);
			if (rT < 0) rT = 0; if (rT > 255)rT = 255;
			cT = Vec3b(0, 255 - rT, rT);
		}
		else
		{
			// BGR
			if (state_state == ResState::INNER) cT = Vec3b(255, 0, 0);
			else if (state_state == ResState::OOB) cT = Vec3b(255, 255, 0);
			else if (state_state == ResState::OUTLIER) cT = Vec3b(0, 0, 255);
			else cT = Vec3b(255, 255, 255);
		}

		for (int it = 0; it < patternNum; ++it)
		{
			if ((projectedTo[it][0] > 2 && projectedTo[it][1] > 2 &&
				projectedTo[it][0] < wG[0] - 3 && projectedTo[it][1] < hG[0] - 3))
				target->debugImage->setPixel1((float)projectedTo[it][0], (float)projectedTo[it][1], cT);
		}
	}

	/// <summary>
	/// 观测残差状态赋值：state_state = state_stateNew；
	/// 若copyJacobians为true：那么需要将前端观测雅可比传递给后端观测雅可比；
	/// </summary>
	/// <param name="copyJacobians">是否拷贝前端雅可比数据</param>
	void PointFrameResidual::applyRes(bool copyJacobians)
	{
		if (copyJacobians)
		{
			// 当前观测超出了图像的范围
			if (state_state == ResState::OOB)
			{
				assert(!efResidual->isActiveAndIsGoodNEW);
				return;	// can never go back from OOB
			}

			// 当前观测在图像范围内并且光度残差在阈值范围之内
			if (state_NewState == ResState::INNER)
			{
				efResidual->isActiveAndIsGoodNEW = true;
				efResidual->takeDataF();
			}
			else
			{
				efResidual->isActiveAndIsGoodNEW = false;
			}
		}

		// 将残差的能量值以及状态置为新计算的值
		setState(state_NewState);
		state_energy = state_NewEnergy;
	}
}