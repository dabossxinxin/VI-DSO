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

#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
	bool EFAdjointsValid = false;		// 视觉信息中相对量相对于绝对量的雅可比是否计算完成
	bool EFIndicesValid = false;		// 视觉信息中关键帧、特征点、观测残差是否准备好
	bool EFDeltaValid = false;			// 视觉信息中相对于线性化点处的增量是否计算完成

	/// <summary>
	/// 计算滑窗优化中惯导信息的Hessian和b 
	/// </summary>
	/// <param name="H"></param>
	/// <param name="b"></param>
	void EnergyFunctional::calcIMUHessian(MatXX& H, VecX& b)
	{
		b = VecX::Zero(7 + nFrames * 15);
		H = MatXX::Zero(7 + nFrames * 15, 7 + nFrames * 15);

		// 两帧图像之间具有IMU测量数据
		if (nFrames == 1) return;

		int imuResCounter = 0;
		double Energy = 0;
		double timeStart = 0, timeEnd = 0, dt = 0;
		int fhIdxS = 0, fhIdxE = 0;
		Vec3 g_w(0, 0, -setting_gravityNorm);

		// TODO：在前端跟踪中已经计算了滑窗关键帧之间的预积分数据，这里可以不用再重复计算一遍
		for (int fhIdx = 0; fhIdx < frames.size() - 1; ++fhIdx)
		{
			fhIdxS = fhIdx;
			fhIdxE = fhIdx + 1;

			// P、V、Q
			VecX r_pvq = VecX::Zero(9);
			MatXX J_pvq = MatXX::Zero(9, 7 + nFrames * 15);

			timeStart = input_picTimestampLeftList[frames[fhIdxS]->data->shell->incomingId];
			timeEnd = input_picTimestampLeftList[frames[fhIdxE]->data->shell->incomingId];
			dt = timeEnd - timeStart;

			imuResCounter++;
			auto Framei = frames[fhIdxS]->data;
			auto Framej = frames[fhIdxE]->data;

			// 计算两帧之间IMU偏置相对残差相对于偏置的Jacobian(陀螺仪偏置和加速度计偏置)
			VecX r_bias = VecX::Zero(6);
			MatXX J_bias = MatXX::Zero(6, 7 + nFrames * 15);

			r_bias.block(0, 0, 3, 1) = Framej->bias_g + Framej->delta_bias_g - (Framei->bias_g + Framei->delta_bias_g);
			r_bias.block(3, 0, 3, 1) = Framej->bias_a + Framej->delta_bias_a - (Framei->bias_a + Framei->delta_bias_a);

			J_bias.block(0, 7 + fhIdxS * 15 + 9, 3, 3) = -Mat33::Identity();	// 偏置残差相对于第i帧陀螺仪偏置的导数
			J_bias.block(0, 7 + fhIdxE * 15 + 9, 3, 3) = Mat33::Identity();		// 偏置残差相对于第j帧陀螺仪偏置的导数
			J_bias.block(3, 7 + fhIdxS * 15 + 12, 3, 3) = -Mat33::Identity();	// 偏置残差相对于第i帧加速度偏置的导数
			J_bias.block(3, 7 + fhIdxE * 15 + 12, 3, 3) = Mat33::Identity();	// 偏置残差相对于第j帧加速度偏置的导数

			// 陀螺仪偏置建模为随机游走,以随机游走的方差的逆作为偏置的权重
			Mat66 covBias = Mat66::Zero();
			covBias.block(0, 0, 3, 3) = GyrRandomWalkNoise * dt;
			covBias.block(3, 3, 3, 3) = AccRandomWalkNoise * dt;
			Mat66 weightBias = Mat66::Identity() * setting_imuWeightNoise * setting_imuWeightNoise * covBias.inverse();
			H += J_bias.transpose() * weightBias * J_bias;
			b += J_bias.transpose() * weightBias * r_bias;

			// 如果两帧之间的时间间隔大于0.5s了放弃此次预积分，长时间的惯导数据保证不了精度
			if (dt > 0.5) continue;

			IMUPreintegrator IMU_preintegrator;
			IMU_preintegrator.reset();
			preintergrate(IMU_preintegrator, Framei->bias_g, Framei->bias_a, timeStart, timeEnd);

			SE3 worldToCam_i = Framei->PRE_worldToCam;
			SE3 worldToCam_j = Framej->PRE_worldToCam;
			SE3 worldToCam_i_evalPT = Framei->get_worldToCam_evalPT();
			SE3 worldToCam_j_evalPT = Framej->get_worldToCam_evalPT();

			// 线性化点上的数据
			Mat44 M_WC_I_PT = T_WD.matrix() * worldToCam_i_evalPT.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_I_PT(M_WC_I_PT * T_BC.inverse().matrix());
			Mat33 R_WB_I_PT = T_WB_I_PT.rotationMatrix();
			Vec3 t_WB_I_PT = T_WB_I_PT.translation();

			Mat44 M_WC_J_PT = T_WD.matrix() * worldToCam_j_evalPT.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_J_PT(M_WC_J_PT * T_BC.inverse().matrix());
			Mat33 R_WB_J_PT = T_WB_J_PT.rotationMatrix();
			Vec3 t_WB_J_PT = T_WB_J_PT.translation();

			Mat44 M_WC_I = T_WD.matrix() * worldToCam_i.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_I(M_WC_I * T_BC.inverse().matrix());
			Mat33 R_WB_I = T_WB_I.rotationMatrix();
			Vec3 t_WB_I = T_WB_I.translation();

			Mat44 M_WC_J = T_WD.matrix() * worldToCam_j.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_J(M_WC_J * T_BC.inverse().matrix());
			Mat33 R_WB_J = T_WB_J.rotationMatrix();
			Vec3 t_WB_J = T_WB_J.translation();

			// 计算IMU预积分P,V,Q残差
			Vec3 r_delta = IMU_preintegrator.getJRBiasg() * Framei->delta_bias_g;
			Mat33 R_delta = SO3::exp(r_delta).matrix();
			Vec3 pre_v = IMU_preintegrator.getDeltaV() + IMU_preintegrator.getJVBiasa() * Framei->delta_bias_a + IMU_preintegrator.getJVBiasg() * Framei->delta_bias_g;
			Vec3 pre_p = IMU_preintegrator.getDeltaP() + IMU_preintegrator.getJPBiasa() * Framei->delta_bias_a + IMU_preintegrator.getJPBiasg() * Framei->delta_bias_g;
			Vec3 res_q = SO3((IMU_preintegrator.getDeltaR() * R_delta).transpose() * R_WB_I.transpose() * R_WB_J).log();
			Vec3 res_v = R_WB_I.transpose() * (Framej->velocity - Framei->velocity - g_w * dt) - pre_v;
			Vec3 res_p = R_WB_I.transpose() * (t_WB_J - t_WB_I - Framei->velocity * dt - 0.5 * g_w * dt * dt) - pre_p;

			Mat99 Cov = IMU_preintegrator.getCovPVPhi();
			
			// 计算惯导数据中残差关于状态量的雅可比
			Mat33 RightJacobianInv_ResR = IMU_preintegrator.JacobianRInv(res_q);
			Mat33 J_resPhi_phi_i = -RightJacobianInv_ResR * R_WB_J.transpose() * R_WB_I;
			Mat33 J_resPhi_phi_j = RightJacobianInv_ResR;
			Mat33 J_resPhi_bg = -RightJacobianInv_ResR * SO3::exp(-res_q).matrix() *
				IMU_preintegrator.JacobianR(r_delta) * IMU_preintegrator.getJRBiasg();

			Mat33 J_resV_phi_i = SO3::hat(R_WB_I.transpose() * (Framej->velocity - Framei->velocity - g_w * dt));
			Mat33 J_resV_v_i = -R_WB_I.transpose();
			Mat33 J_resV_v_j = R_WB_I.transpose();
			Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
			Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();

			Mat33 J_resP_p_i = -Mat33::Identity();
			Mat33 J_resP_p_j = R_WB_I.transpose() * R_WB_J;
			Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
			Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
			Mat33 J_resP_v_i = -R_WB_I.transpose() * dt;
			Mat33 J_resP_phi_i = SO3::hat(R_WB_I.transpose() * (t_WB_J - t_WB_I - Framei->velocity * dt - 0.5 * g_w * dt * dt));

			Mat915 J_imui = Mat915::Zero();//p,q,v,bg,ba;
			J_imui.block(0, 0, 3, 3) = J_resP_p_i;
			J_imui.block(0, 3, 3, 3) = J_resP_phi_i;
			J_imui.block(0, 6, 3, 3) = J_resP_v_i;
			J_imui.block(0, 9, 3, 3) = J_resP_bg;
			J_imui.block(0, 12, 3, 3) = J_resP_ba;

			J_imui.block(3, 3, 3, 3) = J_resPhi_phi_i;
			J_imui.block(3, 9, 3, 3) = J_resPhi_bg;

			J_imui.block(6, 3, 3, 3) = J_resV_phi_i;
			J_imui.block(6, 6, 3, 3) = J_resV_v_i;
			J_imui.block(6, 9, 3, 3) = J_resV_bg;
			J_imui.block(6, 12, 3, 3) = J_resV_ba;

			Mat915 J_imuj = Mat915::Zero();//p,q,v,bg,ba;
			J_imuj.block(0, 0, 3, 3) = J_resP_p_j;
			J_imuj.block(3, 3, 3, 3) = J_resPhi_phi_j;
			J_imuj.block(6, 6, 3, 3) = J_resV_v_j;

			Mat99 Weight = Mat99::Zero();
			Weight.block(0, 0, 3, 3) = Cov.block(0, 0, 3, 3);
			Weight.block(3, 3, 3, 3) = Cov.block(6, 6, 3, 3);
			Weight.block(6, 6, 3, 3) = Cov.block(3, 3, 3, 3);
			Weight = Weight.diagonal().asDiagonal();

			Mat1515 J_reli = Mat1515::Identity();
			Mat1515 J_relj = Mat1515::Identity();

			Mat44 T_tmp_i = T_BC.matrix() * T_WD_l.matrix() * worldToCam_i.matrix();
			J_reli.block(0, 0, 6, 6) = (-1 * Sim3(T_tmp_i).Adj()).block(0, 0, 6, 6);
			Mat44 T_tmp_j = T_BC.matrix() * T_WD_l.matrix() * worldToCam_j.matrix();
			J_relj.block(0, 0, 6, 6) = (-1 * Sim3(T_tmp_j).Adj()).block(0, 0, 6, 6);

			Mat77 J_poseb_wd_i = Sim3(T_tmp_i).Adj() - Sim3(T_BC.matrix() * T_WD_l.matrix()).Adj();
			Mat77 J_poseb_wd_j = Sim3(T_tmp_j).Adj() - Sim3(T_BC.matrix() * T_WD_l.matrix()).Adj();
			J_poseb_wd_i.block(0, 0, 7, 3) = Mat73::Zero();
			J_poseb_wd_j.block(0, 0, 7, 3) = Mat73::Zero();
			// 	J_poseb_wd_i.block(0,3,7,3) = Mat73::Zero();
			// 	J_poseb_wd_j.block(0,3,7,3) = Mat73::Zero();
			// 	J_poseb_wd_i.block(0,6,7,1) = Vec7::Zero();
			// 	J_poseb_wd_j.block(0,6,7,1) = Vec7::Zero();

			if (frames.size() < setting_maxFrames)
			{
				J_poseb_wd_i.block(0, 0, 7, 3) = Mat73::Zero();
				J_poseb_wd_j.block(0, 0, 7, 3) = Mat73::Zero();
				J_poseb_wd_i.block(0, 3, 7, 3) = Mat73::Zero();
				J_poseb_wd_j.block(0, 3, 7, 3) = Mat73::Zero();
				J_poseb_wd_i.block(0, 6, 7, 1) = Vec7::Zero();
				J_poseb_wd_j.block(0, 6, 7, 1) = Vec7::Zero();
			}

			Mat97 J_res_posebi = Mat97::Zero();
			Mat97 J_res_posebj = Mat97::Zero();
			J_res_posebi.block(0, 0, 9, 6) = J_imui.block(0, 0, 9, 6);
			J_res_posebj.block(0, 0, 9, 6) = J_imuj.block(0, 0, 9, 6);

			// 惯导积分计算时使用右雅可比，视觉计算时使用左雅可比，
			Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
			Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
			Mat1515 J_r_l_i = Mat1515::Identity();
			Mat1515 J_r_l_j = Mat1515::Identity();
			J_r_l_i.block(0, 0, 6, 6) = J_xi_r_l_i;
			J_r_l_j.block(0, 0, 6, 6) = J_xi_r_l_j;

			// 计算惯导部分关于世界系与DSO系之间的相似变换的雅可比
			J_pvq.block(0, 0, 9, 7) += J_res_posebi * J_poseb_wd_i;
			J_pvq.block(0, 0, 9, 7) += J_res_posebj * J_poseb_wd_j;
			J_pvq.block(0, 0, 9, 3) = Mat93::Zero();

			J_pvq.block(0, 7 + fhIdxS * 15, 9, 15) += J_imui * J_reli * J_r_l_i;
			J_pvq.block(0, 7 + fhIdxE * 15, 9, 15) += J_imuj * J_relj * J_r_l_j;

			r_pvq.block(0, 0, 3, 1) += res_p;
			r_pvq.block(3, 0, 3, 1) += res_q;
			r_pvq.block(6, 0, 3, 1) += res_v;

			H += (J_pvq.transpose() * Weight * J_pvq);
			b += (J_pvq.transpose() * Weight * r_pvq);

			// 惯导部分的能量值为位置、速度、旋转、偏置部分残差的二范数
			Energy += (r_pvq.transpose() * Weight * r_pvq)[0] + (r_bias.transpose() * weightBias * r_bias)[0];
		}

		// 与视觉部分对应视觉部分的Jacobian同样加上了平移和旋转的尺度参数
		for (int it = 0; it < nFrames; ++it)
		{
			H.block(0, 7 + it * 15, 7 + nFrames * 15, 3) *= SCALE_XI_TRANS;
			H.block(7 + it * 15, 0, 3, 7 + nFrames * 15) *= SCALE_XI_TRANS;

			H.block(0, 7 + it * 15 + 3, 7 + nFrames * 15, 3) *= SCALE_XI_ROT;
			H.block(7 + it * 15 + 3, 0, 3, 7 + nFrames * 15) *= SCALE_XI_ROT;

			b.block(7 + it * 15, 0, 3, 1) *= SCALE_XI_TRANS;
			b.block(7 + it * 15 + 3, 0, 3, 1) *= SCALE_XI_ROT;
		}
	}

	/// <summary>
	/// 系统计算的是光度残差相对于相对位姿以及相对光度参数的雅可比,
	/// 因此要得到绝对位姿和光度参数的解，必须再求相对参数对绝对参数的雅可比
	/// </summary>
	/// <param name="Hcalib">相机内参信息</param>
	void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
	{
		if (adHost != 0) delete[] adHost;
		if (adTarget != 0) delete[] adTarget;
		adHost = new Mat88[nFrames * nFrames];
		adTarget = new Mat88[nFrames * nFrames];

		// target帧序号为行，host帧序号为列
		for (int h = 0; h < nFrames; ++h)
		{
			for (int t = 0; t < nFrames; ++t)
			{
				FrameHessian* host = frames[h]->data;
				FrameHessian* target = frames[t]->data;

				SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

				Mat88 AH = Mat88::Identity();
				Mat88 AT = Mat88::Identity();

				// 这里转置后，后面使用这个矩阵时不用再转置
				AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
				AT.topLeftCorner<6, 6>() = Mat66::Identity();

				Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
				AT(6, 6) = -affLL[0];
				AT(7, 7) = -1;
				AH(6, 6) = affLL[0];
				AH(7, 7) = affLL[0];

				AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
				AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
				AH.block<1, 8>(6, 0) *= SCALE_A;
				AH.block<1, 8>(7, 0) *= SCALE_B;
				AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
				AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
				AT.block<1, 8>(6, 0) *= SCALE_A;
				AT.block<1, 8>(7, 0) *= SCALE_B;

				adHost[h + t * nFrames] = AH;
				adTarget[h + t * nFrames] = AT;
			}
		}

		// 对相机内参添加超强先验 TODO：为什么要加先验信息
		cPrior = VecC::Constant(setting_initialCalibHessian);

		if (adHostF != 0) delete[] adHostF;
		if (adTargetF != 0) delete[] adTargetF;
		adHostF = new Mat88f[nFrames * nFrames];
		adTargetF = new Mat88f[nFrames * nFrames];

		for (int h = 0; h < nFrames; h++)
		{
			for (int t = 0; t < nFrames; t++)
			{
				adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
				adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
			}
		}

		cPriorF = cPrior.cast<float>();
		EFAdjointsValid = true;
	}

	/// <summary>
	/// 优化函数的构造函数：给一些成员变量设置初值
	/// </summary>
	EnergyFunctional::EnergyFunctional()
	{
		adHost = nullptr;
		adTarget = nullptr;

		red = nullptr;

		adHostF = nullptr;
		adTargetF = nullptr;
		adHTdeltaF = nullptr;

		nFrames = nResiduals = nPoints = 0;

		HM_visual = MatXX::Zero(CPARS, CPARS);
		bM_visual = VecX::Zero(CPARS);

		HM_imu = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_imu = VecX::Zero(CPARS + 7);

		HM_imu_half = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_imu_half = VecX::Zero(CPARS + 7);

		HM_bias = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_bias = VecX::Zero(CPARS + 7);

		accSSE_top_L = new AccumulatedTopHessianSSE();
		accSSE_top_A = new AccumulatedTopHessianSSE();
		accSSE_bot = new AccumulatedSCHessianSSE();

		resInA = resInL = resInM = 0;
		currentLambda = 0;
	}

	/// <summary>
	/// 优化函数的析构函数：释放成员内存空间
	/// </summary>
	EnergyFunctional::~EnergyFunctional()
	{
		// 释放滑窗中残差数据、特征数据以及关键帧数据
		for (EFFrame* f : frames)
		{
			for (EFPoint* p : f->points)
			{
				for (EFResidual* r : p->residualsAll)
				{
					r->data->efResidual = nullptr;
					SAFE_DELETE(r);
				}
				p->data->efPoint = nullptr;
				SAFE_DELETE(p);
			}
			f->data->efFrame = nullptr;
			SAFE_DELETE(f);
		}

		// 释放Hessian矩阵累加器handle
		SAFE_DELETE(accSSE_top_L);
		SAFE_DELETE(accSSE_top_A);
		SAFE_DELETE(accSSE_bot);

		// 释放参数相对量对绝对量的雅可比
		SAFE_DELETE(adHost, true);
		SAFE_DELETE(adTarget, true);
		SAFE_DELETE(adHostF, true);
		SAFE_DELETE(adTargetF, true);

		// 释放滑窗两帧之间的相对参数
		SAFE_DELETE(adHTdeltaF, true);
	}

	/// <summary>
	/// 获取关键帧以及所管理特征的参数增量以及滑窗关键帧之间位姿变化、光度变换的增量
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
	{
		// 1、计算滑窗关键帧之间的位姿变化、光度变换参数
		SAFE_DELETE(adHTdeltaF, true);
		adHTdeltaF = new Mat18f[nFrames * nFrames];
		for (int h = 0; h < nFrames; ++h)
		{
			for (int t = 0; t < nFrames; ++t)
			{
				int idx = h + t * nFrames;
				adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					+ frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
			}
		}

		// 2、获取相机内参相对于线性化点处的增量
		cDeltaF = HCalib->value_minus_value_zero.cast<float>();

		// 3、获取滑窗关键帧位姿、光度增量以及所管理的特征逆深度增量
		for (EFFrame* f : frames)
		{
			f->delta = f->data->get_state_minus_stateZero().head<8>();
			f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

			for (EFPoint* p : f->points)
				p->deltaF = p->data->idepth - p->data->idepth_zero;
		}

		EFDeltaValid = true;
	}

	/// <summary>
	/// 将已经激活的特征加入到累加器accSSE_top_A中计算其Hessian和b信息
	/// </summary>
	/// <param name="H">输出Hessian</param>
	/// <param name="b">输出b</param>
	/// <param name="MT">是否多线程操作</param>
	void EnergyFunctional::accumulateAF_MT(MatXX& H, VecX& b, bool MT)
	{
		if (MT)
		{
			red->reduce(std::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, 0, 0);
			red->reduce(std::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>, accSSE_top_A, &allPoints, this,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);
			accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
			resInA = accSSE_top_A->nres[0];
		}
		else
		{
			accSSE_top_A->setZero(nFrames);
			for (EFFrame* f : frames)
				for (EFPoint* p : f->points)
					accSSE_top_A->addPoint<0>(p, this);
			accSSE_top_A->stitchDoubleMT(red, H, b, this, false, false);
			resInA = accSSE_top_A->nres[0];
		}
	}

	/// <summary>
	/// 将已经线性化的特征加入到累加器accSSE_top_L中计算其Hessian和b信息
	/// </summary>
	/// <param name="H">输出Hessian</param>
	/// <param name="b">输出b</param>
	/// <param name="MT">是否多线程操作</param>
	void EnergyFunctional::accumulateLF_MT(MatXX& H, VecX& b, bool MT)
	{
		if (MT)
		{
			red->reduce(std::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, 0, 0);
			red->reduce(std::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>, accSSE_top_L, &allPoints, this,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);
			accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
			resInL = accSSE_top_L->nres[0];
		}
		else
		{
			accSSE_top_L->setZero(nFrames);
			for (EFFrame* f : frames)
				for (EFPoint* p : f->points)
					accSSE_top_L->addPoint<1>(p, this);
			accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
			resInL = accSSE_top_L->nres[0];
		}
	}

	/// <summary>
	/// 将滑窗关键帧中管理的特征加入到累加器accSSE_bot中计算特征的舒尔补信息
	/// </summary>
	/// <param name="H">舒尔补Hessian</param>
	/// <param name="b">舒尔补b</param>
	/// <param name="MT">是否多线程操作</param>
	void EnergyFunctional::accumulateSCF_MT(MatXX& H, VecX& b, bool MT)
	{
		if (MT)
		{
			red->reduce(std::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, 0, 0);
			red->reduce(std::bind(&AccumulatedSCHessianSSE::addPointsInternal, accSSE_bot, &allPoints, true,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);
			accSSE_bot->stitchDoubleMT(red, H, b, this, true);
		}
		else
		{
			accSSE_bot->setZero(nFrames);
			for (EFFrame* f : frames)
				for (EFPoint* p : f->points)
					accSSE_bot->addPoint(p, true);
			accSSE_bot->stitchDoubleMT(red, H, b, this, false);
		}
	}

	/// <summary>
	/// 计算滑窗关键帧管理的特征点在优化中的逆深度更新量
	/// </summary>
	/// <param name="x">滑窗关键帧</param>
	/// <param name="HCalib"></param>
	/// <param name="MT"></param>
	void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
	{
		assert(x.size() == CPARS + nFrames * 8);

		VecXf xF = x.cast<float>();
		Mat18f* xAd = new Mat18f[nFrames * nFrames];

		// 相机内参增量
		HCalib->step = -x.head<CPARS>();
		VecCf cstep = xF.head<CPARS>();

		for (EFFrame* h : frames)
		{
			// 滑窗关键帧位姿、光度参数增量
			h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
			h->data->step.tail<2>().setZero();

			// 滑窗关键帧之间位姿变化、光度变化增量
			for (EFFrame* t : frames)
			{
				xAd[nFrames * h->idx + t->idx] = xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
					+ xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
			}
		}

		// 根据关键帧位姿增量以及相机内参增量计算特征逆深度增量
		if (MT)
			red->reduce(std::bind(&EnergyFunctional::resubstituteFPt, this, cstep, xAd,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);
		else
			resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

		SAFE_DELETE(xAd);
	}

	/// <summary>
	/// 根据滑窗关键帧位姿、光度参数增量以及相机内参增量计算特征逆深度增量
	/// </summary>
	/// <param name="xc">相机内参的更新量</param>
	/// <param name="xAd">滑窗关键帧之间位姿变化、光度变化增量</param>
	/// <param name="min">用于索引优化中的路标点</param>
	/// <param name="max">用于索引优化中的路标点</param>
	/// <param name="stats">状态量信息、当前函数中不使用</param>
	/// <param name="tid">线程ID信息、只有在多线程时调用</param>
	void EnergyFunctional::resubstituteFPt(const VecCf& xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
	{
		for (int k = min; k < max; k++)
		{
			EFPoint* p = allPoints[k];

			// 特征点对应的残差若无效那么这个特征不更新
			int ngoodres = 0;
			/*for (EFResidual* r : p->residualsAll)
				if (r->isActive() && !r->data->stereoResidualFlag) ngoodres++;*/
			for (EFResidual* r : p->residualsAll)
				if (r->isActive()) ngoodres++;
			if (ngoodres == 0)
			{
				p->data->step = 0;
				continue;
			}

			float b = p->bdSumF;
			b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

			for (EFResidual* r : p->residualsAll)
			{
				if (!r->isActive()) continue;
				b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
			}

			p->data->step = -b * p->HdiF;
			if (!std::isfinite(p->data->step))
			{
				printf("INFO: b: %f\n", b);
				printf("INFO: p->HdiF: %f\n", p->HdiF);
				printf("INFO: p->bdSumF: %f\n", p->bdSumF);
				printf("INFO: ngoodres: %d\n", ngoodres);

				printf("INFO: xc: %f,%f,%f,%f\n", xc[0], xc[1], xc[2], xc[3]);
				printf("INFO: p->Hcd_accAF: %f,%f,%f,%f\n", p->Hcd_accAF[0],
					p->Hcd_accAF[1], p->Hcd_accAF[2], p->Hcd_accAF[3]);
				printf("INFO: p->Hcd_accLF: %f,%f,%f,%f\n", p->Hcd_accLF[0],
					p->Hcd_accLF[1], p->Hcd_accLF[2], p->Hcd_accLF[3]);
			}
			assert(std::isfinite(p->data->step));
		}
	}

	/// <summary>
	/// TODO：计算视觉边缘化能量值
	/// </summary>
	/// <returns></returns>
	double EnergyFunctional::calcMEnergyF()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		VecX delta = getStitchedDeltaF();
		return delta.dot(2 * bM_visual + HM_visual * delta);
	}

	/// <summary>
	/// TODO：计算滑窗关键帧中特征光度残差构成的能量值
	/// </summary>
	/// <param name="min">特征索引</param>
	/// <param name="max">特征索引</param>
	/// <param name="stats">多线程内返回需传出的状态值</param>
	/// <param name="tid">线程ID</param>
	void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
	{
		Accumulator11 E;
		E.initialize();
		VecCf dc = cDeltaF;

		for (int it = min; it < max; ++it)
		{
			auto p = allPoints[it];
			float dd = p->deltaF;

			for (EFResidual* r : p->residualsAll)
			{
				// 计算线性化观测残差的能量值
				if (!r->isLinearized || !r->isActive()) continue;

				Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
				RawResidualJacobian* rJ = r->J;

				// compute Jp*delta
				float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
					+ rJ->Jpdc[0].dot(dc)
					+ rJ->Jpdd[0] * dd;

				float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
					+ rJ->Jpdc[1].dot(dc)
					+ rJ->Jpdd[1] * dd;

				__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
				__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
				__m128 delta_a = _mm_set1_ps((float)(dp[6]));
				__m128 delta_b = _mm_set1_ps((float)(dp[7]));

				// E = (2*r+J*delta)*J*delta.
				for (int it = 0; it + 3 < patternNum; it += 4)
				{
					__m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx)) + it), Jp_delta_x);
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx + 1)) + it), Jp_delta_y));
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF)) + it), delta_a));
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF + 1)) + it), delta_b));

					__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF) + it);
					r0 = _mm_add_ps(r0, r0);
					r0 = _mm_add_ps(r0, Jdelta);
					Jdelta = _mm_mul_ps(Jdelta, r0);
					E.updateSSENoShift(Jdelta);
				}
				for (int it = ((patternNum >> 2) << 2); it < patternNum; ++it)
				{
					float Jdelta = rJ->JIdx[0][it] * Jp_delta_x_1 + rJ->JIdx[1][it] * Jp_delta_y_1 +
						rJ->JabF[0][it] * dp[6] + rJ->JabF[1][it] * dp[7];
					E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[it])));
				}
			}
			E.updateSingle(p->deltaF * p->deltaF * p->priorF);		// 最后把每个特征的先验也加上
		}
		E.finish();
		(*stats)[0] += E.A;
	}

	/// <summary>
	/// TODO：计算滑窗优化中迭代优化后优化问题能量值
	/// </summary>
	/// <returns>优化问题能量值</returns>
	double EnergyFunctional::calcLEnergyF_MT()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		// 1、计算滑窗关键帧状态的先验能量值
		double E = 0;
		for (EFFrame* f : frames)
			E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

		// 2、计算相机内参的先验能量值
		E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

		// 3、计算滑窗中管理的特征的能量值
		red->reduce(std::bind(&EnergyFunctional::calcLEnergyPt, this,
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);

		return E + red->stats[0];
	}

	/// <summary>
	/// 向滑窗优化函数中添加观测残差
	/// </summary>
	/// <param name="r">观测残差在前端中的表示</param>
	/// <returns>观测残差在后端中的表示</returns>
	EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
	{
		EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
		efr->idxInAll = r->point->efPoint->residualsAll.size();
		r->point->efPoint->residualsAll.emplace_back(efr);

		if (efr->data->stereoResidualFlag == false)
			connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

		nResiduals++;
		r->efResidual = efr;

		return efr;
	}

	/// <summary>
	/// 向滑窗优化函数中添加关键帧优化参数
	/// </summary>
	/// <param name="fh">关键帧在前端中的表示</param>
	/// <param name="Hcalib">相机内参信息</param>
	/// <returns>关键帧在后端中的表示</returns>
	EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
	{
		// 1、添加关键帧到滑窗优化中
		EFFrame* eff = new EFFrame(fh);
		eff->idx = frames.size();
		frames.emplace_back(eff);

		nFrames++;
		fh->efFrame = eff;

		EFFrame* effRight = new EFFrame(fh->frameRight);
		effRight->idx = frames.size() + 10000;
		fh->frameRight->efFrame = effRight;

		// 2、新帧进来后，扩展视觉舒尔补信息矩阵维度
		assert(HM_visual.cols() == 8 * nFrames + CPARS - 8);
		bM_visual.conservativeResize(8 * nFrames + CPARS);
		HM_visual.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
		bM_visual.tail<8>().setZero();
		HM_visual.rightCols<8>().setZero();
		HM_visual.bottomRows<8>().setZero();

		// 3、新帧进来后，扩展惯导舒尔补信息矩阵维度
		bM_imu.conservativeResize(17 * nFrames + CPARS + 7);
		HM_imu.conservativeResize(17 * nFrames + CPARS + 7, 17 * nFrames + CPARS + 7);
		bM_imu.tail<17>().setZero();
		HM_imu.rightCols<17>().setZero();
		HM_imu.bottomRows<17>().setZero();

		bM_bias.conservativeResize(17 * nFrames + CPARS + 7);
		HM_bias.conservativeResize(17 * nFrames + CPARS + 7, 17 * nFrames + CPARS + 7);
		bM_bias.tail<17>().setZero();
		HM_bias.rightCols<17>().setZero();
		HM_bias.bottomRows<17>().setZero();

		bM_imu_half.conservativeResize(17 * nFrames + CPARS + 7);
		HM_imu_half.conservativeResize(17 * nFrames + CPARS + 7, 17 * nFrames + CPARS + 7);
		bM_imu_half.tail<17>().setZero();
		HM_imu_half.rightCols<17>().setZero();
		HM_imu_half.bottomRows<17>().setZero();

		EFIndicesValid = false;
		EFAdjointsValid = false;
		EFDeltaValid = false;

		// 4、重新计算滑窗关键帧之间相对位姿对绝对位姿的导数
		setAdjointsF(Hcalib);
		makeIDX();

		// 5、TODO：更新滑窗关键帧之间的共视关系
		for (EFFrame* fh2 : frames)
		{
			connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
			if (fh2 != eff)
				connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
		}

		return eff;
	}

	/// <summary>
	/// 向滑窗优化函数中添加特征优化参数
	/// </summary>
	/// <param name="ph">特征在前端中的表示</param>
	/// <returns>特征在后端中的表示</returns>
	EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
	{
		EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
		efp->idxInPoints = ph->host->efFrame->points.size();
		ph->host->efFrame->points.emplace_back(efp);

		nPoints++;
		ph->efPoint = efp;

		EFIndicesValid = false;

		return efp;
	}

	/// <summary>
	/// 边缘化惯导信息
	/// </summary>
	/// <param name="fh">待边缘化的图像帧</param>
	void EnergyFunctional::marginalizeFrame_imu(EFFrame* fh)
	{
		if (nFrames >= setting_maxFrames)
			setting_imuTrackReady = true;

		// 相机内参、DSO系与Metric系变换、p、q、v、bg、ba、a、b
		MatXX HM_change = MatXX::Zero(CPARS + 7 + nFrames * 17, CPARS + 7 + nFrames * 17);
		VecX bM_change = VecX::Zero(CPARS + 7 + nFrames * 17);

		MatXX HM_change_half = MatXX::Zero(CPARS + 7 + nFrames * 17, CPARS + 7 + nFrames * 17);
		VecX bM_change_half = VecX::Zero(CPARS + 7 + nFrames * 17);

        Vec3 g_w(0,0,-setting_gravityNorm);
		double mar_weight = 0.5;
		double timeStart = 0, timeEnd = 0, dt = 0;

		// 对于视觉中边缘化的关键帧，在IMU中也需要对其进行边缘化，并同时维护两种边缘化的惯导信息
		for (int fhIdx = fh->idx - 1; fhIdx < fh->idx + 1; ++fhIdx)
		{
			if (fhIdx < 0) continue;

			MatXX J_all = MatXX::Zero(9, CPARS + 7 + nFrames * 17);
			MatXX J_all_half = MatXX::Zero(9, CPARS + 7 + nFrames * 17);
			VecX r_all = VecX::Zero(9);

			timeStart = input_picTimestampLeftList[frames[fhIdx]->data->shell->incomingId];
			timeEnd = input_picTimestampLeftList[frames[fhIdx + 1]->data->shell->incomingId];
			dt = timeEnd - timeStart;

			if (dt > 0.5)
            {
                printf("WARNING: integration time > 0.5, continue!\n");
                continue;
            }

			auto Framei = frames[fhIdx]->data;
			auto Framej = frames[fhIdx + 1]->data;

			SE3 worldToCam_i_evalPT = Framei->get_worldToCam_evalPT();	// 线性化点处的第i帧的位姿
			SE3 worldToCam_j_evalPT = Framej->get_worldToCam_evalPT();	// 线性化点处的第j帧的位姿
			SE3 worldToCam_i = Framei->PRE_worldToCam;
			SE3 worldToCam_j = Framej->PRE_worldToCam;

            IMUPreintegrator IMU_preintegrator;
            IMU_preintegrator.reset();
            preintergrate(IMU_preintegrator,Framei->bias_g, Framei->bias_a, timeStart, timeEnd);

			Mat44 M_WC_I = T_WD.matrix() * worldToCam_i.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_I(M_WC_I * T_BC.inverse().matrix());
			Mat33 R_WB_I = T_WB_I.rotationMatrix();
			Vec3 t_WB_I = T_WB_I.translation();

			Mat44 M_WC_J = T_WD.matrix() * worldToCam_j.inverse().matrix() * T_WD.inverse().matrix();
			SE3 T_WB_J(M_WC_J * T_BC.inverse().matrix());
			Mat33 R_WB_J = T_WB_J.rotationMatrix();
			Vec3 t_WB_J = T_WB_J.translation();

			Vec3 delta_v = IMU_preintegrator.getJVBiasa() * Framei->delta_bias_a + IMU_preintegrator.getJVBiasg() * Framei->delta_bias_g;
			Vec3 delta_p = IMU_preintegrator.getJPBiasa() * Framei->delta_bias_a + IMU_preintegrator.getJPBiasg() * Framei->delta_bias_g;
			Vec3 delta_q = IMU_preintegrator.getJRBiasg() * Framei->delta_bias_g;
			Vec3 res_q = SO3((IMU_preintegrator.getDeltaR() * SO3::exp(delta_q).matrix()).transpose() * R_WB_I.transpose() * R_WB_J).log();
			Vec3 res_v = R_WB_I.transpose() * (Framej->velocity - Framei->velocity - g_w * dt) - (IMU_preintegrator.getDeltaV() + delta_v);
			Vec3 res_p = R_WB_I.transpose() * (t_WB_J - t_WB_I - Framei->velocity * dt - 0.5 * g_w * dt * dt) - (IMU_preintegrator.getDeltaP() + delta_p);

			Mat99 Cov = IMU_preintegrator.getCovPVPhi();

			Mat33 J_resPhi_phi_i = -IMU_preintegrator.JacobianRInv(res_q) * R_WB_J.transpose() * R_WB_I;
			Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_q);
			Mat33 J_resPhi_bg = -IMU_preintegrator.JacobianRInv(res_q) * SO3::exp(-res_q).matrix() *
				IMU_preintegrator.JacobianR(delta_q) * IMU_preintegrator.getJRBiasg();

			Mat33 J_resV_phi_i = SO3::hat(R_WB_I.transpose() * (Framej->velocity - Framei->velocity - g_w * dt));
			Mat33 J_resV_v_i = -R_WB_I.transpose();
			Mat33 J_resV_v_j = R_WB_I.transpose();
			Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
			Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();

			Mat33 J_resP_p_i = -Mat33::Identity();
			Mat33 J_resP_p_j = R_WB_I.transpose() * R_WB_J;
			Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
			Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
			Mat33 J_resP_v_i = -R_WB_I.transpose() * dt;
			Mat33 J_resP_phi_i = SO3::hat(R_WB_I.transpose() * (t_WB_J - t_WB_I - Framei->velocity * dt - 0.5 * g_w * dt * dt));

            // 计算惯导残差对第i帧状态的雅可比
			Mat915 J_imui = Mat915::Zero(); //p,q,v,bias_g,bias_a;
			J_imui.block(0, 0, 3, 3) = J_resP_p_i;
			J_imui.block(0, 3, 3, 3) = J_resP_phi_i;
			J_imui.block(0, 6, 3, 3) = J_resP_v_i;
			J_imui.block(0, 9, 3, 3) = J_resP_bg;
			J_imui.block(0, 12, 3, 3) = J_resP_ba;

			J_imui.block(3, 3, 3, 3) = J_resPhi_phi_i;
			J_imui.block(3, 9, 3, 3) = J_resPhi_bg;

			J_imui.block(6, 3, 3, 3) = J_resV_phi_i;
			J_imui.block(6, 6, 3, 3) = J_resV_v_i;
			J_imui.block(6, 9, 3, 3) = J_resV_bg;
			J_imui.block(6, 12, 3, 3) = J_resV_ba;

            // 计算惯导残差对第j帧状态的雅可比
			Mat915 J_imuj = Mat915::Zero();
			J_imuj.block(0, 0, 3, 3) = J_resP_p_j;
			J_imuj.block(3, 3, 3, 3) = J_resPhi_phi_j;
			J_imuj.block(6, 6, 3, 3) = J_resV_v_j;

			Mat99 Weight = Mat99::Zero();
			Weight.block(0, 0, 3, 3) = Cov.block(0, 0, 3, 3);
			Weight.block(3, 3, 3, 3) = Cov.block(6, 6, 3, 3);
			Weight.block(6, 6, 3, 3) = Cov.block(3, 3, 3, 3);

            Mat99 weightTmp = Mat99::Zero();
            for (int idx = 0; idx < 9; ++idx)
                weightTmp(idx,idx) = Weight(idx,idx);
			Weight = setting_imuWeightNoise * setting_imuWeightNoise * weightTmp.inverse();
            if (haveNanData(Weight,9,9))
            {
                printf("WARNING: imu data weight is nan!\n");
                exit(-1);
            }

			Mat1515 J_reli = Mat1515::Identity();
			Mat1515 J_relj = Mat1515::Identity();
			Mat44 T_tmp_i = T_BC.matrix() * T_WD_l.matrix() * worldToCam_i.matrix();
			Mat44 T_tmp_j = T_BC.matrix() * T_WD_l.matrix() * worldToCam_j.matrix();
			J_reli.block(0, 0, 6, 6) = (-1 * Sim3(T_tmp_i).Adj()).block(0, 0, 6, 6);
			J_relj.block(0, 0, 6, 6) = (-1 * Sim3(T_tmp_j).Adj()).block(0, 0, 6, 6);

			Mat1515 J_reli_half = Mat1515::Identity();
			Mat1515 J_relj_half = Mat1515::Identity();
			Mat44 T_tmpi_half = T_BC.matrix() * T_WD_l_half.matrix() * worldToCam_i.matrix();
			Mat44 T_tmpj_half = T_BC.matrix() * T_WD_l_half.matrix() * worldToCam_j.matrix();
			J_reli_half.block(0, 0, 6, 6) = (-1 * Sim3(T_tmpi_half).Adj()).block(0, 0, 6, 6);
			J_relj_half.block(0, 0, 6, 6) = (-1 * Sim3(T_tmpj_half).Adj()).block(0, 0, 6, 6);

			Mat77 J_poseb_wd_i = Sim3(T_tmp_i).Adj() - Sim3(T_BC.matrix() * T_WD_l.matrix()).Adj();
			Mat77 J_poseb_wd_j = Sim3(T_tmp_j).Adj() - Sim3(T_BC.matrix() * T_WD_l.matrix()).Adj();
			Mat77 J_poseb_wd_i_half = Sim3(T_tmpi_half).Adj() - Sim3(T_BC.matrix() * T_WD_l_half.matrix()).Adj();
			Mat77 J_poseb_wd_j_half = Sim3(T_tmpj_half).Adj() - Sim3(T_BC.matrix() * T_WD_l_half.matrix()).Adj();
			J_poseb_wd_i.block(0, 0, 7, 3) = Mat73::Zero();
			J_poseb_wd_j.block(0, 0, 7, 3) = Mat73::Zero();
			J_poseb_wd_i_half.block(0, 0, 7, 3) = Mat73::Zero();
			J_poseb_wd_j_half.block(0, 0, 7, 3) = Mat73::Zero();

			Mat97 J_res_posebi = Mat97::Zero();
			Mat97 J_res_posebj = Mat97::Zero();
			J_res_posebi.block(0, 0, 9, 6) = J_imui.block(0, 0, 9, 6);
			J_res_posebj.block(0, 0, 9, 6) = J_imuj.block(0, 0, 9, 6);

			Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
			Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
			Mat1515 J_r_l_i = Mat1515::Identity();
			Mat1515 J_r_l_j = Mat1515::Identity();
			J_r_l_i.block(0, 0, 6, 6) = J_xi_r_l_i;
			J_r_l_j.block(0, 0, 6, 6) = J_xi_r_l_j;

			J_all.block(0, CPARS, 9, 7) += J_res_posebi * J_poseb_wd_i;
			J_all.block(0, CPARS, 9, 7) += J_res_posebj * J_poseb_wd_j;
			J_all.block(0, CPARS, 9, 3) = Mat93::Zero();

			J_all.block(0, CPARS + 7 + fhIdx * 17, 9, 6) += J_imui.block(0, 0, 9, 6) * J_reli.block(0, 0, 6, 6) * J_xi_r_l_i;
			J_all.block(0, CPARS + 7 + (fhIdx + 1) * 17, 9, 6) += J_imuj.block(0, 0, 9, 6) * J_relj.block(0, 0, 6, 6) * J_xi_r_l_j;
			J_all.block(0, CPARS + 7 + fhIdx * 17 + 8, 9, 9) += J_imui.block(0, 6, 9, 9);
			J_all.block(0, CPARS + 7 + (fhIdx + 1) * 17 + 8, 9, 9) += J_imuj.block(0, 6, 9, 9);

			J_all_half.block(0, CPARS, 9, 7) += J_res_posebi * J_poseb_wd_i_half;
			J_all_half.block(0, CPARS, 9, 7) += J_res_posebj * J_poseb_wd_j_half;
			J_all_half.block(0, CPARS, 9, 3) = Mat93::Zero();

			J_all_half.block(0, CPARS + 7 + fhIdx * 17, 9, 6) += J_imui.block(0, 0, 9, 6) * J_reli_half.block(0, 0, 6, 6) * J_xi_r_l_i;
			J_all_half.block(0, CPARS + 7 + (fhIdx + 1) * 17, 9, 6) += J_imuj.block(0, 0, 9, 6) * J_relj_half.block(0, 0, 6, 6) * J_xi_r_l_j;
			J_all_half.block(0, CPARS + 7 + fhIdx * 17 + 8, 9, 9) += J_imui.block(0, 6, 9, 9);
			J_all_half.block(0, CPARS + 7 + (fhIdx + 1) * 17 + 8, 9, 9) += J_imuj.block(0, 6, 9, 9);

			r_all.block(0, 0, 3, 1) += res_p;
			r_all.block(3, 0, 3, 1) += res_q;
			r_all.block(6, 0, 3, 1) += res_v;

			HM_change += (J_all.transpose() * Weight * J_all);
			bM_change += (J_all.transpose() * Weight * r_all);

			HM_change_half = (J_all_half.transpose() * Weight * J_all_half);
			bM_change_half = (J_all_half.transpose() * Weight * r_all);
			//HM_change_half = HM_change_half * setting_margWeightFacImu;
			//bM_change_half = bM_change_half * setting_margWeightFacImu;

            // 计算惯导偏置信息对应的雅可比和残差
			VecX r_all_bias = VecX::Zero(6);
			MatXX J_all_bias = MatXX::Zero(6, CPARS + 7 + nFrames * 17);
			r_all_bias.block(0, 0, 3, 1) = Framej->bias_g + Framej->delta_bias_g - (Framei->bias_g + Framei->delta_bias_g);
			r_all_bias.block(3, 0, 3, 1) = Framej->bias_a + Framej->delta_bias_a - (Framei->bias_a + Framei->delta_bias_a);

			J_all_bias.block(0, CPARS + 7 + fhIdx * 17 + 8 + 3, 3, 3) = -Mat33::Identity();
			J_all_bias.block(0, CPARS + 7 + (fhIdx + 1) * 17 + 8 + 3, 3, 3) = Mat33::Identity();
			J_all_bias.block(3, CPARS + 7 + fhIdx * 17 + 8 + 6, 3, 3) = -Mat33::Identity();
			J_all_bias.block(3, CPARS + 7 + (fhIdx + 1) * 17 + 8 + 6, 3, 3) = Mat33::Identity();

			Mat66 Cov_bias = Mat66::Zero();
			Cov_bias.block(0, 0, 3, 3) = GyrRandomWalkNoise * dt;
			Cov_bias.block(3, 3, 3, 3) = AccRandomWalkNoise * dt;
			Mat66 weightBias = Mat66::Identity() * setting_imuWeightNoise * setting_imuWeightNoise * Cov_bias.inverse();
			HM_bias += (J_all_bias.transpose() * weightBias * J_all_bias * setting_margWeightFacImu);
			bM_bias += (J_all_bias.transpose() * weightBias * r_all_bias * setting_margWeightFacImu);
		}

		HM_change = HM_change * setting_margWeightFacImu;
		bM_change = bM_change * setting_margWeightFacImu;

		HM_change_half = HM_change_half * setting_margWeightFacImu;
		bM_change_half = bM_change_half * setting_margWeightFacImu;

		// 获取视觉部分得到的滑窗内优化量增量信息
		VecX StitchedDelta = getStitchedDeltaF();
		VecX delta_b = VecX::Zero(CPARS + 7 + nFrames * 17);

		// 将获取的视觉部分得到关键帧位姿增量赋值给待边缘化帧的前后两帧
		for (int fhIdx = fh->idx - 1; fhIdx < fh->idx + 1; ++fhIdx)
		{
			if (fhIdx < 0) continue;

			timeStart = input_picTimestampLeftList[frames[fhIdx]->data->shell->incomingId];
			timeEnd = input_picTimestampLeftList[frames[fhIdx + 1]->data->shell->incomingId];
			if (timeEnd - timeStart > 0.5) continue;

			if (fhIdx == fh->idx - 1)
			{
				delta_b.block(CPARS + 7 + 17 * fhIdx, 0, 6, 1) = StitchedDelta.block(CPARS + fhIdx * 8, 0, 6, 1);
				frames[fhIdx]->m_flag = true;
			}
			if (fhIdx == fh->idx)
			{
				delta_b.block(CPARS + 7 + 17 * (fhIdx + 1), 0, 6, 1) = StitchedDelta.block(CPARS + (fhIdx + 1) * 8, 0, 6, 1);
				frames[fhIdx + 1]->m_flag = true;
			}
		}

		delta_b.block(CPARS, 0, 7, 1) = Sim3(T_WD_l.inverse() * T_WD).log();

		VecX delta_b_half = delta_b;
		delta_b_half.block(CPARS, 0, 7, 1) = Sim3(T_WD_l_half.inverse() * T_WD).log();

        // 去掉视觉部分更新出来的信息量
		bM_change -= HM_change * delta_b;
		bM_change_half -= HM_change_half * delta_b_half;

		double s_now = T_WD.scale(), di = 1;
		if (s_last > s_now) di = (s_last + 0.001) / s_now;
		else di = (s_now + 0.001) / s_last;
		s_last = s_now;

		if (di > d_now) d_now = di;
		if (d_now > setting_dynamicMin) d_now = setting_dynamicMin;

		printf("INFO: s_now: %f, s_middle: %f\n", s_now, s_middle);
		printf("INFO: d_now: %f, scale_l: %f\n", d_now, T_WD_l.scale());

		if (di > d_half) d_half = di;
		if (d_half > setting_dynamicMin) d_half = setting_dynamicMin;
		bool side = s_now > s_middle;

		// 对应论文中if upper != lastUpper
		if (side != side_last || marg_num_half == 0)
		{
			// 清空尺度相关的信息量
			HM_imu_half.block(CPARS + 6, 0, 1, HM_imu_half.cols()) = MatXX::Zero(1, HM_imu_half.cols());
			HM_imu_half.block(0, CPARS + 6, HM_imu_half.rows(), 1) = MatXX::Zero(HM_imu_half.rows(), 1);
			bM_imu_half[CPARS + 6] = 0;

			//HM_imu_half.setZero();
			//bM_imu_half.setZero();
			d_half = di;
			if (d_half > setting_dynamicMin) d_half = setting_dynamicMin;

			marg_num_half = 0;
			T_WD_l_half = T_WD;
		}

		marg_num_half++;
		side_last = side;

		HM_imu_half += HM_change_half;
		bM_imu_half += bM_change_half;

        // TODO：为什么惯导偏置信息要和IMU的其他信息分开边缘化
		schurHessianImu(fh, HM_imu_half, bM_imu_half);
		schurHessianImu(fh, HM_bias, bM_bias);

		// TODO：是否意味着尺度信息已经收敛了
		if (marg_num > 25) setting_useDynamicMargin = false;

		if ((s_now > s_middle * d_now || s_now < s_middle / d_now) && setting_useDynamicMargin)
		{
			HM_imu = HM_imu_half;
			bM_imu = bM_imu_half;
			//s_middle = s_middle/d_now;
			s_middle = s_now;
			//d_now = d_half;
			marg_num = marg_num_half;

			//HM_imu_half.setZero();
			//bM_imu_half.setZero();
			HM_imu_half.block(CPARS + 6, 0, 1, HM_imu_half.cols()) = MatXX::Zero(1, HM_imu_half.cols());
			HM_imu_half.block(0, CPARS + 6, HM_imu_half.rows(), 1) = MatXX::Zero(HM_imu_half.rows(), 1);
			bM_imu_half[CPARS + 6] = 0;

			d_half = di;
			if (d_half > setting_dynamicMin) d_half = setting_dynamicMin;
			marg_num_half = 0;
			T_WD_l = T_WD_l_half;
			state_twd = Sim3(T_WD_l.inverse() * T_WD).log();
		}
		else
        {
            HM_imu += HM_change;
            bM_imu += bM_change;
            marg_num++;
            schurHessianImu(fh, HM_imu, bM_imu);
        }
	}

    void EnergyFunctional::schurHessianImu(EFFrame* fh, MatXX& H, VecX& b)
    {
        int ndim = nFrames * 17 + CPARS + 7 - 17;   // new dimension
        int odim = nFrames * 17 + CPARS + 7;        // old dimension

        if ((int)fh->idx != (int)frames.size() - 1)
        {
            int io = fh->idx * 17 + CPARS + 7;
            int ntail = 17 * (nFrames - fh->idx - 1);
            assert((io + 17 + ntail) == nFrames * 17 + CPARS + 7);

            Vec17 bTmp = b.segment<17>(io);
            VecX tailTMP = b.tail(ntail);
            b.segment(io, ntail) = tailTMP;
            b.tail<17>() = bTmp;

            MatXX HtmpCol = H.block(0, io, odim, 17);
            MatXX rightColsTmp = H.rightCols(ntail);
            H.block(0, io, odim, ntail) = rightColsTmp;
            H.rightCols(17) = HtmpCol;

            MatXX HtmpRow = H.block(io, 0, 17, odim);
            MatXX botRowsTmp = H.bottomRows(ntail);
            H.block(io, 0, ntail, odim) = botRowsTmp;
            H.bottomRows(17) = HtmpRow;
        }

        VecX SVec = (H.diagonal().cwiseAbs() + VecX::Constant(H.cols(), 10)).cwiseSqrt();
        VecX SVecI = SVec.cwiseInverse();

        MatXX HMScaled = SVecI.asDiagonal() * H * SVecI.asDiagonal();
        VecX bMScaled = SVecI.asDiagonal() * b;

        Mat1717 hpi = HMScaled.bottomRightCorner<17, 17>();
        hpi = 0.5f * (hpi + hpi);
        hpi = hpi.inverse();
        hpi = 0.5f * (hpi + hpi);
        if (haveNanData(hpi,17,17)) hpi = Mat1717::Zero();

        MatXX bli = HMScaled.bottomLeftCorner(17, ndim).transpose() * hpi;
        HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17, ndim);
        bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<17>();

        HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
        bMScaled = SVec.asDiagonal() * bMScaled;

        H = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
        b = bMScaled.head(ndim);
    }

    void EnergyFunctional::schurHessianVisual(EFFrame* fh, MatXX& H, VecX& b)
    {
        int ndim = nFrames * 8 + CPARS - 8;		// new dimension
        int odim = nFrames * 8 + CPARS;			// old dimension

        if ((int)fh->idx != (int)frames.size() - 1)
        {
            int io = fh->idx * 8 + CPARS;
            int ntail = 8 * (nFrames - fh->idx - 1);
            assert((io + 8 + ntail) == nFrames * 8 + CPARS);

            Vec8 bTmp = b.segment<8>(io);
            VecX tailTmp = b.tail(ntail);
            b.segment(io, ntail) = tailTmp;
            b.tail<8>() = bTmp;

            MatXX HtmpCol = H.block(0, io, odim, 8);
            MatXX rightColsTmp = H.rightCols(ntail);
            H.block(0, io, odim, ntail) = rightColsTmp;
            H.rightCols(8) = HtmpCol;

            MatXX HtmpRow = H.block(io, 0, 8, odim);
            MatXX botRowsTmp = H.bottomRows(ntail);
            H.block(io, 0, ntail, odim) = botRowsTmp;
            H.bottomRows(8) = HtmpRow;
        }

        // 边缘化关键帧之前，边缘化帧的先验信息需要加上去，否则后面这个信息就丢失了
        H.bottomRightCorner<8, 8>().diagonal() += fh->prior;
        b.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

        VecX SVec = (H.diagonal().cwiseAbs() + VecX::Constant(H.cols(), 10)).cwiseSqrt();
        VecX SVecI = SVec.cwiseInverse();

        MatXX HMScaled = SVecI.asDiagonal() * H * SVecI.asDiagonal();
        VecX bMScaled = SVecI.asDiagonal() * b;

        Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
        hpi = 0.5f * (hpi + hpi);
        hpi = hpi.inverse();
        hpi = 0.5f * (hpi + hpi);
        if (haveNanData(hpi,8,8)) hpi = Mat88::Zero();

        MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
        HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
        bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

        HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
        bMScaled = SVec.asDiagonal() * bMScaled;

        H = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
        b = bMScaled.head(ndim);
    }

	/// <summary>
	/// 滑窗优化中边缘化前端中标记为边缘化的关键帧
	/// </summary>
	/// <param name="fh">待边缘化的帧在后端中的表示</param>
	void EnergyFunctional::marginalizeFrame(EFFrame* fh)
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);
		assert(fh->points.empty());

		// 1、惯导信息边缘化
		if (setting_useImu)
			marginalizeFrame_imu(fh);

		// 2、视觉信息边缘化
        schurHessianVisual(fh, HM_visual, bM_visual);

		// 3、去除滑窗关键帧序列中的待边缘化的关键帧
		for (unsigned int it = fh->idx; it + 1 < frames.size(); it++)
		{
			frames[it] = frames[it + 1];
			frames[it]->idx = it;
		}

		nFrames--;
		frames.pop_back();
		fh->data->efFrame = nullptr;

		EFIndicesValid = false;
		EFAdjointsValid = false;
		EFDeltaValid = false;

		makeIDX();
		SAFE_DELETE(fh);
	}

	/// <summary>
	/// 滑窗优化中边缘化标记为边缘化状态的特征；
	/// 边缘化关键帧时，关键帧中包含很多路标点信息，若直接丢弃则优化问题中
	/// 会损失很多信息，此时的做法是将有效的路标点收集起来进行边缘化求解Hessian
	/// </summary>
	void EnergyFunctional::marginalizePointsF()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		allPointsToMarg.clear();

		// 1、收集滑窗关键帧中标记为PS_MARGINALIZE且有效残差数量大于0的特征
		for (EFFrame* f : frames)
		{
			for (int it = 0; it < (int)f->points.size(); ++it)
			{
				EFPoint* p = f->points[it];
				if (p->stateFlag == EFPointStatus::PS_MARGINALIZE)
				{
					p->priorF *= setting_idepthFixPriorMargFac;
					for (EFResidual* r : p->residualsAll)
						if (r->isActive() && !r->data->stereoResidualFlag)
							connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;

					int ngoodres = 0;
					for (EFResidual* r : p->residualsAll)
						if (r->isActive() && !r->data->stereoResidualFlag) ngoodres++;
					if (ngoodres > 0)
						allPointsToMarg.emplace_back(p);	// 收集可被观测并标记为边缘化点的特征
					else
						removePoint(p);						// 不可被观测的特征直接在滑窗优化中移除
				}
			}
		}

		// 2、使用边缘化特征计算Hessian以及b，为了包含质量较好的特征，需要计算舒尔补项
		accSSE_bot->setZero(nFrames);
		accSSE_top_A->setZero(nFrames);

		for (EFPoint* p : allPointsToMarg)
		{
			accSSE_top_A->addPoint<2>(p, this);
			accSSE_bot->addPoint(p, false);
			removePoint(p);
		}

		MatXX M, Msc;
		VecX Mb, Mbsc;
		accSSE_top_A->stitchDouble(M, Mb, this, false, false);
		accSSE_bot->stitchDouble(Msc, Mbsc, this);

		resInM += accSSE_top_A->nres[0];

		MatXX H = M - Msc;
		VecX b = Mb - Mbsc;

		if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
		{
			bool haveFirstFrame = false;
			for (EFFrame* f : frames)
				if (f->frameID == 0) haveFirstFrame = true;

			if (!haveFirstFrame)
				orthogonalize(&b, &H);
		}

		HM_visual += setting_margWeightFac * H;
		bM_visual += setting_margWeightFac * b;

		if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
			orthogonalize(&bM_visual, &HM_visual);

		EFIndicesValid = false;
		makeIDX();
	}

	/// <summary>
	/// 删除滑窗优化中的观测残差：找到残差对应的特征，在特征管理的残差序列中删除该残差
	/// </summary>
	/// <param name="r">观测残差在滑窗优化中的表示</param>
	void EnergyFunctional::dropResidual(EFResidual* r)
	{
		EFPoint* p = r->point;
		assert(r == p->residualsAll[r->idxInAll]);

		p->residualsAll[r->idxInAll] = p->residualsAll.back();
		p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
		p->residualsAll.pop_back();

		// 删除残差时，统计当前残差对应的主帧中有效残差和无效残差的数量
		if (r->isActive())
			r->host->data->shell->statistics_goodResOnThis++;
		else
			r->host->data->shell->statistics_outlierResOnThis++;

		if (!r->data->stereoResidualFlag)
			connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;

		nResiduals--;
		r->data->efResidual = nullptr;
		SAFE_DELETE(r);
	}

	/// <summary>
	/// 删除滑窗优化中状态为PS_DROP的特征
	/// </summary>
	void EnergyFunctional::dropPointsF()
	{
		for (EFFrame* f : frames)
		{
			for (int it = 0; it < (int)f->points.size(); ++it)
			{
				EFPoint* p = f->points[it];
				if (p->stateFlag == EFPointStatus::PS_DROP)
				{
					removePoint(p);
					it--;
				}
			}
		}

		EFIndicesValid = false;
		makeIDX();
	}

	/// <summary>
	/// 删除滑窗优化中的指定特征
	/// </summary>
	/// <param name="p">特征在后端中的表示形式</param>
	void EnergyFunctional::removePoint(EFPoint* p)
	{
		// 1、删除滑窗优化中该特征中管理的所有观测残差
		for (EFResidual* r : p->residualsAll)
			dropResidual(r);

		// 2、在该特征对应的主帧管理的特征中删除该特征
		EFFrame* h = p->host;
		h->points[p->idxInPoints] = h->points.back();
		h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
		h->points.pop_back();

		nPoints--;
		p->data->efPoint = nullptr;

		EFIndicesValid = false;
		SAFE_DELETE(p);
	}

	/// <summary>
	/// 正交化Hessian以及b，去除零空间分量的干扰
	/// </summary>
	/// <param name="b">输入b矩阵</param>
	/// <param name="H">输入Hessian</param>
	void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
	{
		// 1、将零空间向量进行列合并组成零空间矩阵
		std::vector<VecX> ns;
		ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
		ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
		/*if(setting_affineOptModeA <= 0)
			ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
		if(setting_affineOptModeB <= 0)
			ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());*/

		MatXX N(ns[0].rows(), ns.size());
		for (unsigned int it = 0; it < ns.size(); ++it)
			N.col(it) = ns[it].normalized();

		// 2、求解零空间的伪逆，使用SVD方法直接求解，也可以使用Npi := N * (N' * N)^-1，但是N' * N不一定可逆
		Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX SNN = svdNN.singularValues();
		double minSv = 1e10, maxSv = 0;
		for (int it = 0; it < SNN.size(); ++it)
		{
			if (SNN[it] < minSv) minSv = SNN[it];
			if (SNN[it] > maxSv) maxSv = SNN[it];
		}
		for (int it = 0; it < SNN.size(); ++it)
		{
			if (SNN[it] <= setting_solverModeDelta * maxSv) SNN[it] = 0;
			else SNN[it] = 1.0 / SNN[it];
		}

		MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
		MatXX NNpiT = N * Npi.transpose(); 	// [dim] x [dim].
		MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

		// 3、将信息量向零空间投影，并减掉投影在零空间的信息量
		if (b != nullptr) *b -= NNpiTS * *b;
		if (H != nullptr) *H -= NNpiTS * *H * NNpiTS;
	}

	/// <summary>
	/// 滑窗优化执行函数，求解滑窗关键帧优化后状态量
	/// </summary>
	/// <param name="iteration">优化迭代次数</param>
	/// <param name="lambda">优化阻尼因子</param>
	/// <param name="HCalib">相机内参信息</param>
	void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
	{
		if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
		if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		MatXX HL_visual, HA_visual, Hsc_visual, HFinal_visual;
		VecX  bL_visual, bA_visual, bsc_visual, bFinal_visual;

		MatXX H_VisualAndImu = MatXX::Zero(CPARS + 7 + 17 * nFrames, CPARS + 7 + 17 * nFrames);
		VecX b_VisualAndImu = VecX::Zero(CPARS + 7 + 17 * nFrames);

		MatXX H_imu;
		VecX b_imu;
		calcIMUHessian(H_imu, b_imu);

		accumulateAF_MT(HA_visual, bA_visual, setting_multiThreading);
		accumulateLF_MT(HL_visual, bL_visual, setting_multiThreading);
		accumulateSCF_MT(Hsc_visual, bsc_visual, setting_multiThreading);

		// 边缘化信息bM中添加最新状态更新量的影响
		VecX StitchedDeltaVisual = getStitchedDeltaF();
		VecX StitchedDeltaIMU = VecX::Zero(CPARS + 7 + nFrames * 17);
		for (int idx = 0; idx < nFrames; ++idx)
		{
			if (frames[idx]->m_flag)
				StitchedDeltaIMU.block(CPARS + 7 + 17 * idx, 0, 6, 1) = StitchedDeltaVisual.block(CPARS + 8 * idx, 0, 6, 1);
		}
		StitchedDeltaIMU.block(CPARS, 0, 7, 1) = state_twd;

		VecX bM_deltaVisual = (bM_visual + HM_visual * StitchedDeltaVisual);
		VecX bM_deltaImu = (bM_imu + HM_imu * StitchedDeltaIMU);

		if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
		{
			bool haveFirstFrame = false;
			for (EFFrame* f : frames)
				if (f->frameID == 0) haveFirstFrame = true;

			MatXX HT_act = HL_visual + HA_visual - Hsc_visual;
			VecX bT_act = bL_visual + bA_visual - bsc_visual;

			if (!haveFirstFrame)
				orthogonalize(&bT_act, &HT_act);

			HFinal_visual = HT_act + HM_visual;
			bFinal_visual = bT_act + bM_deltaVisual;

			lastHS = HFinal_visual;
			lastbS = bFinal_visual;

			for (int idx = 0; idx < 8 * nFrames + CPARS; ++idx)
				HFinal_visual(idx, idx) *= (1 + lambda);
		}
		else
		{
			HFinal_visual = HL_visual + HA_visual + HM_visual - Hsc_visual;
			bFinal_visual = bL_visual + bA_visual + bM_deltaVisual - bsc_visual;

			lastHS = HFinal_visual;
			lastbS = bFinal_visual;

			for (int idx = 0; idx < 8 * nFrames + CPARS; idx++)
				HFinal_visual(idx, idx) *= (1 + lambda);
		}

		H_imu(6, 6) += setting_initialScaleHessian;
		H_imu.block(3, 3, 3, 3) += setting_initialIMUHessian * Mat33::Identity();

		for (int idx = 0; idx < nFrames; ++idx)
		{
			H_imu.block(7 + 15 * idx + 9, 7 + 15 * idx + 9, 3, 3) += setting_initialbgHessian * Mat33::Identity();
			H_imu.block(7 + 15 * idx + 12, 7 + 15 * idx + 12, 3, 3) += setting_initialbaHessian * Mat33::Identity();
		}
		for (int idx = 0; idx < 7 + 15 * nFrames; ++idx)
			H_imu(idx, idx) *= (1 + lambda);

		// 将单独的惯导信息和视觉信息写入同一个信息矩阵中
		// c:相机内参，t:世界系与DSO系变换，pi:帧位姿和光度，v:帧速度，bg:陀螺仪偏置，ba:加速度偏置
		H_VisualAndImu.block(0, 0, CPARS, CPARS) = HFinal_visual.block(0, 0, CPARS, CPARS);	// Hcc
		H_VisualAndImu.block(CPARS, CPARS, 7, 7) = H_imu.block(0, 0, 7, 7);					// Htt
		b_VisualAndImu.block(0, 0, CPARS, 1) = bFinal_visual.block(0, 0, CPARS, 1);			// bc
		b_VisualAndImu.block(CPARS, 0, 7, 1) = b_imu.block(0, 0, 7, 1);						// bt

		for (int idx = 0; idx < nFrames; ++idx)
		{
			// Hc_pi和Hpi_c：c为相机内参，pi为帧位姿
			H_VisualAndImu.block(0, CPARS + 7 + idx * 17, CPARS, 8) += HFinal_visual.block(0, CPARS + idx * 8, CPARS, 8);
			H_VisualAndImu.block(CPARS + 7 + idx * 17, 0, 8, CPARS) += HFinal_visual.block(CPARS + idx * 8, 0, 8, CPARS);

			// Ht_pi和Hpi_t：t为世界系与DSO系变换
			H_VisualAndImu.block(CPARS, CPARS + 7 + idx * 17, 7, 6) += H_imu.block(0, 7 + idx * 15, 7, 6);
			H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS, 6, 7) += H_imu.block(7 + idx * 15, 0, 6, 7);

			// Ht_v,Ht_bg,Ht_ba 和 Hv_t,Hbg_t,Hba_t
			H_VisualAndImu.block(CPARS, CPARS + 7 + idx * 17 + 8, 7, 9) += H_imu.block(0, 7 + idx * 15 + 6, 7, 9);
			H_VisualAndImu.block(CPARS + 7 + idx * 17 + 8, CPARS, 9, 7) += H_imu.block(7 + idx * 15 + 6, 0, 9, 7);

			// Hpi_pi，视觉和惯导信息中都有这条信息
			H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17, 8, 8) += HFinal_visual.block(CPARS + idx * 8, CPARS + idx * 8, 8, 8);
			H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17, 6, 6) += H_imu.block(7 + idx * 15, 7 + idx * 15, 6, 6);

			// Hs_s、Hs_pi、Hpi_s，s为v、bg、ba三者统称，这里的pi去除了光度参数
			H_VisualAndImu.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + idx * 17 + 8, 9, 9) += H_imu.block(7 + idx * 15 + 6, 7 + idx * 15 + 6, 9, 9);
			H_VisualAndImu.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + idx * 17, 9, 6) += H_imu.block(7 + idx * 15 + 6, 7 + idx * 15, 9, 6);
			H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17 + 8, 6, 9) += H_imu.block(7 + idx * 15, 7 + idx * 15 + 6, 6, 9);

			// pi、v、ba、bg每一帧关键帧都有
			for (int j = idx + 1; j < nFrames; ++j)
			{
				// Hpi_pj 和 Hpj_pi，pi和pj表示不同关键帧位姿，包含光度参数
				H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17, 8, 8) += HFinal_visual.block(CPARS + idx * 8, CPARS + j * 8, 8, 8);
				H_VisualAndImu.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17, 8, 8) += HFinal_visual.block(CPARS + j * 8, CPARS + idx * 8, 8, 8);

				// Hpi_pj 和 Hpj_pi，pi和pj表示不同关键帧位姿，不包含光度参数
				H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17, 6, 6) += H_imu.block(7 + idx * 15, 7 + j * 15, 6, 6);
				H_VisualAndImu.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17, 6, 6) += H_imu.block(7 + j * 15, 7 + idx * 15, 6, 6);

				// Hsi_sj 和 Hsj_si，s为v、bg、ba三者统称，i和j表示不同关键帧
				H_VisualAndImu.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + j * 17 + 8, 9, 9) += H_imu.block(7 + idx * 15 + 6, 7 + j * 15 + 6, 9, 9);
				H_VisualAndImu.block(CPARS + 7 + j * 17 + 8, CPARS + 7 + idx * 17 + 8, 9, 9) += H_imu.block(7 + j * 15 + 6, 7 + idx * 15 + 6, 9, 9);

				// Hsi_pj，si为第i帧状态，pj为第j帧状态
				H_VisualAndImu.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + j * 17, 9, 6) += H_imu.block(7 + idx * 15 + 6, 7 + j * 15, 9, 6);
				H_VisualAndImu.block(CPARS + 7 + j * 17 + 8, CPARS + 7 + idx * 17, 9, 6) += H_imu.block(7 + j * 15 + 6, 7 + idx * 15, 9, 6);

				// Hpi_sj。pi为第i帧状态，sj为第j帧状态
				H_VisualAndImu.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17 + 8, 6, 9) += H_imu.block(7 + idx * 15, 7 + j * 15 + 6, 6, 9);
				H_VisualAndImu.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17 + 8, 6, 9) += H_imu.block(7 + j * 15, 7 + idx * 15 + 6, 6, 9);
			}

			// bpi、bs，s 为v、bg、ba三者统称
			b_VisualAndImu.block(CPARS + 7 + 17 * idx, 0, 8, 1) += bFinal_visual.block(CPARS + 8 * idx, 0, 8, 1);
			b_VisualAndImu.block(CPARS + 7 + 17 * idx, 0, 6, 1) += b_imu.block(7 + 15 * idx, 0, 6, 1);
			b_VisualAndImu.block(CPARS + 7 + 17 * idx + 8, 0, 9, 1) += b_imu.block(7 + 15 * idx + 6, 0, 9, 1);
		}

		H_VisualAndImu += (HM_imu + HM_bias);
		b_VisualAndImu += (bM_deltaImu + bM_bias);

		VecX x_visual = VecX::Zero(CPARS + 8 * nFrames);
		VecX x_visualAndImu = VecX::Zero(CPARS + 7 + 17 * nFrames);

		if (setting_solverMode & SOLVER_SVD)
		{
			VecX SVecI = HFinal_visual.diagonal().cwiseSqrt().cwiseInverse();
			MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_visual * SVecI.asDiagonal();
			VecX bFinalScaled = SVecI.asDiagonal() * bFinal_visual;
			Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

			VecX S = svd.singularValues();
			double minSv = 1e10, maxSv = 0;
			for (int it = 0; it < S.size(); ++it)
			{
				if (S[it] < minSv) minSv = S[it];
				if (S[it] > maxSv) maxSv = S[it];
			}

			VecX Ub = svd.matrixU().transpose() * bFinalScaled;
			int setZero = 0;
			for (int it = 0; it < Ub.size(); ++it)
			{
				if (S[it] < setting_solverModeDelta * maxSv)
				{
					Ub[it] = 0;
					setZero++;
				}

				if ((setting_solverMode & SOLVER_SVD_CUT7) && (it >= Ub.size() - 7))
				{
					Ub[it] = 0;
					setZero++;
				}
				else
					Ub[it] /= S[it];
			}
			x_visual = SVecI.asDiagonal() * svd.matrixV() * Ub;
		}
		else
		{
			// 判断是否使用惯导数据参与优化计算
			if (!setting_useImu)
			{
				VecX SVecI = (HFinal_visual.diagonal() + VecX::Constant(HFinal_visual.cols(), 10)).cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_visual * SVecI.asDiagonal();
				x_visual = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_visual);
			}
			else
			{
				VecX SVecI = (H_VisualAndImu.diagonal() + VecX::Constant(H_VisualAndImu.cols(), 10)).cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * H_VisualAndImu * SVecI.asDiagonal();
				x_visualAndImu = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * b_VisualAndImu);

				x_visual.block(0, 0, CPARS, 1) = x_visualAndImu.block(0, 0, CPARS, 1);
				for (int it = 0; it < nFrames; ++it)
				{
					x_visual.block(CPARS + it * 8, 0, 8, 1) = x_visualAndImu.block(CPARS + 7 + 17 * it, 0, 8, 1);
					frames[it]->data->step_imu = -x_visualAndImu.block(CPARS + 7 + 17 * it + 8, 0, 9, 1);
				}
				step_twd = -x_visualAndImu.block(CPARS, 0, 7, 1);

				if (!std::isfinite(x_visual[0]))
				{
					printf("ERROR: catch a error in build hessian!\n");
				}
			}
		}

		if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
		{
			VecX xOld = x_visual;
			orthogonalize(&x_visual, 0);
		}

        // 判断解是否有效，若解无效则Hessian数据保存出来
        if (std::isnan(x_visual[0]) || std::isnan(x_visual[1]) ||
        std::isnan(x_visual[2]) || std::isnan(x_visual[3]))
        {
            char bufVisual[100];
			snprintf(bufVisual, 100, "./visual_hessian_%d.txt", nFrames * 8 + CPARS);
			saveDataTxt(HFinal_visual, bufVisual, nFrames * 8 + CPARS, nFrames * 8 + CPARS);

            char bufImu[100];
			snprintf(bufImu, 100, "./imu_hessian_%d.txt", 7 + nFrames * 15);
			saveDataTxt(H_imu, bufImu, 7 + nFrames * 15, 7 + nFrames * 15);
			
			char bufMImu[100];
			snprintf(bufMImu, 100, "./marg_imu_hessian_%d.txt", 7 + nFrames * 15);
			saveDataTxt(HM_imu, bufMImu, 7 + nFrames * 15, 7 + nFrames * 15);

			char bufMBias[100];
			snprintf(bufMBias, 100, "./marg_bias_hessian_%d.txt", 7 + nFrames * 15);
			saveDataTxt(HM_bias, bufMBias, 7 + nFrames * 15, 7 + nFrames * 15);

            char bufTotal[100];
			snprintf(bufTotal, 100, "./total_hessian_%d.txt", 7 + nFrames * 17 + CPARS);
			saveDataTxt(H_VisualAndImu, bufTotal, 7 + nFrames * 17 + CPARS, 7 + nFrames * 17 + CPARS);
        }

		lastX = x_visual;
		currentLambda = lambda;
		resubstituteF_MT(x_visual, HCalib, setting_multiThreading);
		currentLambda = 0;
	}

	/// <summary>
	/// 滑窗中关键帧或特征点或残差有变动时，需要重新编号
	/// </summary>
	void EnergyFunctional::makeIDX()
	{
		for (unsigned int idx = 0; idx < frames.size(); ++idx)
			frames[idx]->idx = idx;

		allPoints.clear();

		for (EFFrame* f : frames)
		{
			for (EFPoint* p : f->points)
			{
				allPoints.emplace_back(p);
				for (EFResidual* r : p->residualsAll)
				{
					r->hostIDX = r->host->idx;
					r->targetIDX = r->target->idx;
					if (r->data->stereoResidualFlag)
						r->targetIDX = frames[frames.size() - 1]->idx;
				}
			}
		}

		EFIndicesValid = true;
	}

	/// <summary>
	/// 获取滑窗关键帧相机内参、相机姿态以及相机光度参数增量
	/// </summary>
	/// <returns>滑窗关键帧内参、位姿以及光度参数增量</returns>
	VecX EnergyFunctional::getStitchedDeltaF() const
	{
		VecX d = VecX(CPARS + nFrames * 8);
		d.head<CPARS>() = cDeltaF.cast<double>();
		for (int h = 0; h < nFrames; ++h)
			d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
		return d;
	}
}