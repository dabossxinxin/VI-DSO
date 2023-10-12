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
	bool EFAdjointsValid = false;
	bool EFIndicesValid = false;
	bool EFDeltaValid = false;

	void EnergyFunctional::calcIMUHessian(MatXX& H, VecX& b)
	{
		b = VecX::Zero(7 + nFrames * 15);
		H = MatXX::Zero(7 + nFrames * 15, 7 + nFrames * 15);

		// 两帧图像之间具有IMU测量数据
		if (nFrames == 1) return;

		int imuResCounter = 0;
		int imuStartStamp = 0;
		double Energy = 0;
		double timeStart = 0, timeEnd = 0;
		double dt = 0, delta_t = 0;
		int fhIdxS = 0, fhIdxE = 0;

		Vec3 g_w;
		g_w << 0, 0, -G_norm;

		for (int fhIdx = 0; fhIdx < frames.size() - 1; ++fhIdx)
		{
			fhIdxS = fhIdx;
			fhIdxE = fhIdx + 1;

			// P、V、Q
			VecX r_pvq = VecX::Zero(9);
			MatXX J_pvq = MatXX::Zero(9, 7 + nFrames * 15);

			IMUPreintegrator IMU_preintegrator;
			IMU_preintegrator.reset();

			timeStart = pic_time_stamp[frames[fhIdxS]->data->shell->incoming_id];
			timeEnd = pic_time_stamp[frames[fhIdxE]->data->shell->incoming_id];
			dt = timeEnd - timeStart;

			imuResCounter++;
			FrameHessian* Framei = frames[fhIdxS]->data;
			FrameHessian* Framej = frames[fhIdxE]->data;

			// 计算两帧之间IMU偏置相对残差相对于偏置的Jacobian
			VecX r_bias = VecX::Zero(6);
			MatXX J_bias = MatXX::Zero(6, 7 + nFrames * 15);

			r_bias.block(0, 0, 3, 1) = Framej->bias_g + Framej->delta_bias_g - (Framei->bias_g + Framei->delta_bias_g);
			r_bias.block(3, 0, 3, 1) = Framej->bias_a + Framej->delta_bias_a - (Framei->bias_a + Framei->delta_bias_a);

			J_bias.block(0, 7 + fhIdxS * 15 + 9, 3, 3) = -Mat33::Identity();
			J_bias.block(0, 7 + fhIdxE * 15 + 9, 3, 3) = Mat33::Identity();
			J_bias.block(3, 7 + fhIdxS * 15 + 12, 3, 3) = -Mat33::Identity();
			J_bias.block(3, 7 + fhIdxE * 15 + 12, 3, 3) = Mat33::Identity();

			// 陀螺仪偏置建模为随机游走,以随机游走的方差的逆作为偏置的权重
			Mat66 covBias = Mat66::Zero();
			covBias.block(0, 0, 3, 3) = GyrRandomWalkNoise * dt;
			covBias.block(3, 3, 3, 3) = AccRandomWalkNoise * dt;
			Mat66 weightBias = Mat66::Identity() * imu_weight * imu_weight * covBias.inverse();
			H += J_bias.transpose() * weightBias * J_bias;
			b += J_bias.transpose() * weightBias * r_bias;

			// 如果两帧之间的时间间隔大于0.5s了放弃此次预积分
			// 原因是IMU测量在较短时间内可保证一定精度
			if (dt > 0.5) continue;

			// TODO:这里时刻应该精准对齐，也就是说需要对IMU的测量进行插值
			// 搜索当前帧时刻对应的IMU数据索引
			imuStartStamp = 0;
			for (int tIdx = 0; tIdx < imu_time_stamp.size(); ++tIdx)
			{
				if (imu_time_stamp[tIdx] > timeStart ||
					std::fabs(timeStart - imu_time_stamp[tIdx]) < 0.001)
				{
					imuStartStamp = tIdx;
					break;
				}
			}

			// 从当前帧时刻到下一帧时刻内的IMU测量值预积分
			while (true)
			{
				if (imu_time_stamp[imuStartStamp + 1] < timeEnd)
					delta_t = imu_time_stamp[imuStartStamp + 1] - imu_time_stamp[imuStartStamp];
				else
				{
					delta_t = timeEnd - imu_time_stamp[imuStartStamp];
					if (delta_t < 0.000001) break;
				}

				IMU_preintegrator.update(m_gry[imuStartStamp] - Framei->bias_g, m_acc[imuStartStamp] - Framei->bias_a, delta_t);
				if (imu_time_stamp[imuStartStamp + 1] >= timeEnd) break;
				imuStartStamp++;
			}

			SE3 worldToCam_i = Framei->PRE_worldToCam;
			SE3 worldToCam_j = Framej->PRE_worldToCam;
			SE3 worldToCam_i_evalPT = Framei->get_worldToCam_evalPT();
			SE3 worldToCam_j_evalPT = Framej->get_worldToCam_evalPT();

			//利用伴随性质将坐标系转换到世界系中 
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

			Mat915 J_imuj = Mat915::Zero();
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

			Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
			Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
			Mat1515 J_r_l_i = Mat1515::Identity();
			Mat1515 J_r_l_j = Mat1515::Identity();
			J_r_l_i.block(0, 0, 6, 6) = J_xi_r_l_i;
			J_r_l_j.block(0, 0, 6, 6) = J_xi_r_l_j;

			// 这里将位置部分置为零那么雅可比只对尺度部分有影响
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

	void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
	{
		if (adHost != 0) delete[] adHost;
		if (adTarget != 0) delete[] adTarget;
		adHost = new Mat88[nFrames * nFrames];
		adTarget = new Mat88[nFrames * nFrames];

		for (int h = 0; h < nFrames; h++)
		{
			for (int t = 0; t < nFrames; t++)
			{
				FrameHessian* host = frames[h]->data;
				FrameHessian* target = frames[t]->data;

				SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

				Mat88 AH = Mat88::Identity();
				Mat88 AT = Mat88::Identity();

				AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
				AT.topLeftCorner<6, 6>() = Mat66::Identity();

				Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
				AT(6, 6) = -affLL[0];
				AH(6, 6) = affLL[0];
				AT(7, 7) = -1;
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

	EnergyFunctional::EnergyFunctional()
	{
		adHost = 0;
		adTarget = 0;

		red = 0;

		adHostF = 0;
		adTargetF = 0;
		adHTdeltaF = 0;

		nFrames = nResiduals = nPoints = 0;

		HM = MatXX::Zero(CPARS, CPARS);
		bM = VecX::Zero(CPARS);

		HM_imu = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_imu = VecX::Zero(CPARS + 7);

		HM_bias = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_bias = VecX::Zero(CPARS + 7);

		HM_imu_half = MatXX::Zero(CPARS + 7, CPARS + 7);
		bM_imu_half = VecX::Zero(CPARS + 7);

		accSSE_top_L = new AccumulatedTopHessianSSE();
		accSSE_top_A = new AccumulatedTopHessianSSE();
		accSSE_bot = new AccumulatedSCHessianSSE();

		resInA = resInL = resInM = 0;
		currentLambda = 0;
	}

	EnergyFunctional::~EnergyFunctional()
	{
		for (EFFrame* f : frames)
		{
			for (EFPoint* p : f->points)
			{
				for (EFResidual* r : p->residualsAll)
				{
					r->data->efResidual = 0;
					delete r;
				}
				p->data->efPoint = 0;
				delete p; p = NULL;
			}
			f->data->efFrame = 0;
			delete f; f = NULL;
		}

		if (adHost != 0) delete[] adHost;
		if (adTarget != 0) delete[] adTarget;
		if (adHostF != 0) delete[] adHostF;
		if (adTargetF != 0) delete[] adTargetF;
		if (adHTdeltaF != 0) delete[] adHTdeltaF;

		delete accSSE_top_L; accSSE_top_L = NULL;
		delete accSSE_top_A; accSSE_top_A = NULL;
		delete accSSE_bot; accSSE_bot = NULL;
	}

	void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
	{
		if (adHTdeltaF != 0) delete[] adHTdeltaF;
		adHTdeltaF = new Mat18f[nFrames * nFrames];
		for (int h = 0; h < nFrames; h++)
		{
			for (int t = 0; t < nFrames; t++)
			{
				int idx = h + t * nFrames;
				adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					+ frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
			}
		}

		cDeltaF = HCalib->value_minus_value_zero.cast<float>();
		for (EFFrame* f : frames)
		{
			f->delta = f->data->get_state_minus_stateZero().head<8>();
			f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

			for (EFPoint* p : f->points)
				p->deltaF = p->data->idepth - p->data->idepth_zero;
		}

		EFDeltaValid = true;
	}

	// accumulates & shifts L.
	void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
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

	// accumulates & shifts L.
	void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
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

	void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
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

	// x：相机姿态以及内参负更新量
	// Hcalib：相机相机内参信息
	// MT：是否开启多线程的标志
	void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
	{
		assert(x.size() == CPARS + nFrames * 8);

		VecXf xF = x.cast<float>();			// 相机姿态以及内参负更新量
		Mat18f* xAd = new Mat18f[nFrames*nFrames];

		HCalib->step = -x.head<CPARS>();	// 相机内参更新量
		VecCf cstep = xF.head<CPARS>();		// 相机内参负更新量

		for (EFFrame* h : frames)
		{
			h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);	// 相机相对姿态更新量
			h->data->step.tail<2>().setZero();

			for (EFFrame* t : frames)
			{
				xAd[nFrames * h->idx + t->idx] = xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
					+ xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
			}
		}

		if (MT)
			red->reduce(std::bind(&EnergyFunctional::resubstituteFPt, this, cstep, xAd, 
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);
		else
			resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

		delete[] xAd;
		xAd = nullptr;
	}
	
	/// <summary>
	/// 更新特征点的逆深度信息
	/// </summary>
	/// <param name="xc">相机内参的更新量</param>
	/// <param name="xAd"></param>
	/// <param name="min">用于索引优化中的路标点</param>
	/// <param name="max">用于索引优化中的路标点</param>
	/// <param name="stats">状态量信息、当前函数中不使用</param>
	/// <param name="tid">线程ID信息、只有在多线程时调用</param>
	void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
	{
		for (int k = min; k < max; k++)
		{
			EFPoint* p = allPoints[k];

			// 特征点对应的残差若无效那么这个特征不更新
			int ngoodres = 0;
			//for(EFResidual* r : p->residualsAll) if(r->isActive()&&r->data->stereoResidualFlag==false) ngoodres++;
			for(EFResidual* r : p->residualsAll) if (r->isActive()) ngoodres++;
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
				b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
			}

			p->data->step = -b * p->HdiF;
			if (std::isfinite(p->data->step) == false)
			{
				LOG(INFO) << "b: " << b;
				LOG(INFO) << "p->HdiF: " << p->HdiF;
				LOG(INFO) << "p->bdSumF: " << p->bdSumF;
				LOG(INFO) << "xc: " << xc.transpose();
				LOG(INFO) << "p->Hcd_accAF: " << p->Hcd_accAF.transpose() << " p->Hcd_accLF: " << p->Hcd_accLF.transpose();
				LOG(INFO) << "ngoodres: " << ngoodres;
			}
			assert(std::isfinite(p->data->step));
		}
	}

	double EnergyFunctional::calcMEnergyF()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		VecX delta = getStitchedDeltaF();
		return delta.dot(2 * bM + HM * delta);
	}

	void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
	{
		Accumulator11 E;
		E.initialize();
		VecCf dc = cDeltaF;

		for (int i = min; i < max; i++)
		{
			EFPoint* p = allPoints[i];
			float dd = p->deltaF;

			for (EFResidual* r : p->residualsAll)
			{
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

				for (int i = 0; i + 3 < patternNum; i += 4)
				{
					// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
					__m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx)) + i), Jp_delta_x);
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx + 1)) + i), Jp_delta_y));
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF)) + i), delta_a));
					Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF + 1)) + i), delta_b));

					__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF) + i);
					r0 = _mm_add_ps(r0, r0);
					r0 = _mm_add_ps(r0, Jdelta);
					Jdelta = _mm_mul_ps(Jdelta, r0);
					E.updateSSENoShift(Jdelta);
				}
				for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
				{
					float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
						rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
					E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
				}
			}
			E.updateSingle(p->deltaF*p->deltaF*p->priorF);
		}
		E.finish();
		(*stats)[0] += E.A;
	}

	double EnergyFunctional::calcLEnergyF_MT()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		double E = 0;
		for (EFFrame* f : frames)
			E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

		E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

		red->reduce(std::bind(&EnergyFunctional::calcLEnergyPt, this, 
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, allPoints.size(), 50);

		return E + red->stats[0];
	}

	EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
	{
		EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
		efr->idxInAll = r->point->efPoint->residualsAll.size();
		r->point->efPoint->residualsAll.push_back(efr);
		if (efr->data->stereoResidualFlag == false)
			connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

		nResiduals++;
		r->efResidual = efr;

		return efr;
	}

	EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
	{
		EFFrame* eff = new EFFrame(fh);
		eff->idx = frames.size();
		frames.push_back(eff);

		nFrames++;
		fh->efFrame = eff;

		//stereo
		EFFrame* eff_right = new EFFrame(fh->frame_right);
		eff_right->idx = frames.size() + 10000;
		// 	eff_right->idx = frames.size();
		fh->frame_right->efFrame = eff_right;

		assert(HM.cols() == 8 * nFrames + CPARS - 8);
		bM.conservativeResize(8 * nFrames + CPARS);
		HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
		bM.tail<8>().setZero();
		HM.rightCols<8>().setZero();
		HM.bottomRows<8>().setZero();

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

		setAdjointsF(Hcalib);
		makeIDX();

		for (EFFrame* fh2 : frames)
		{
			connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
			if (fh2 != eff)
				connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
		}

		return eff;
	}

	EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
	{
		EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
		efp->idxInPoints = ph->host->efFrame->points.size();
		ph->host->efFrame->points.push_back(efp);

		nPoints++;
		ph->efPoint = efp;

		EFIndicesValid = false;

		return efp;
	}

	void EnergyFunctional::dropResidual(EFResidual* r)
	{
		EFPoint* p = r->point;
		assert(r == p->residualsAll[r->idxInAll]);

		p->residualsAll[r->idxInAll] = p->residualsAll.back();
		p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
		p->residualsAll.pop_back();

		if (r->isActive())
			r->host->data->shell->statistics_goodResOnThis++;
		else
			r->host->data->shell->statistics_outlierResOnThis++;

		if (!r->data->stereoResidualFlag)
			connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;

		nResiduals--;
		r->data->efResidual = 0;
		delete r; r = NULL;
	}

	void EnergyFunctional::marginalizeFrame_imu(EFFrame* fh)
	{
		int ndim = nFrames * 17 + CPARS + 7 - 17;	// 新的惯导Hessian矩阵维度
		int odim = nFrames * 17 + CPARS + 7;		// 旧的惯导Hessian矩阵维度

		if (nFrames >= setting_maxFrames)
			imu_track_ready = true;

		// 相机内参、DSO系与Metric系变换、p、q、v、bg、ba、a、b
		MatXX HM_change = MatXX::Zero(CPARS + 7 + nFrames * 17, CPARS + 7 + nFrames * 17);
		VecX bM_change = VecX::Zero(CPARS + 7 + nFrames * 17);

		MatXX HM_change_half = MatXX::Zero(CPARS + 7 + nFrames * 17, CPARS + 7 + nFrames * 17);
		VecX bM_change_half = VecX::Zero(CPARS + 7 + nFrames * 17);

		int imuStartIdx = 0;
		double mar_weight = 0.5;
		double time_start = 0, time_end = 0, dt = 0;

		// 对于视觉中边缘化的关键帧，在IMU中也需要对其进行边缘化
		for (int fhIdx = fh->idx - 1; fhIdx < fh->idx + 1; ++fhIdx)
		{
			if (fhIdx < 0) continue;

			MatXX J_all = MatXX::Zero(9, CPARS + 7 + nFrames * 17);
			MatXX J_all_half = MatXX::Zero(9, CPARS + 7 + nFrames * 17);
			VecX r_all = VecX::Zero(9);
			IMUPreintegrator IMU_preintegrator;
			IMU_preintegrator.reset();

			time_start = pic_time_stamp[frames[fhIdx]->data->shell->incoming_id];
			time_end = pic_time_stamp[frames[fhIdx + 1]->data->shell->incoming_id];
			dt = time_end - time_start;

			if (dt > 0.5) continue;

			FrameHessian* Framei = frames[fhIdx]->data;
			FrameHessian* Framej = frames[fhIdx + 1]->data;

			SE3 worldToCam_i_evalPT = Framei->get_worldToCam_evalPT();	// 线性化点处的第i帧的位姿
			SE3 worldToCam_j_evalPT = Framej->get_worldToCam_evalPT();	// 线性化点处的第j帧的位姿
			SE3 worldToCam_i = Framei->PRE_worldToCam;
			SE3 worldToCam_j = Framej->PRE_worldToCam;

			for (int idx = 0; idx < imu_time_stamp.size(); ++idx)
			{
				if (imu_time_stamp[idx] > time_start ||
					std::fabs(time_start - imu_time_stamp[idx]) < 0.001)
				{
					imuStartIdx = idx;
					break;
				}
			}

			while (1) 
			{
				double delta_t;
				if (imu_time_stamp[imuStartIdx + 1] < time_end)
					delta_t = imu_time_stamp[imuStartIdx + 1] - imu_time_stamp[imuStartIdx];
				else 
				{
					delta_t = time_end - imu_time_stamp[imuStartIdx];
					if (delta_t < 0.000001) break;
				}
				IMU_preintegrator.update(m_gry[imuStartIdx] - Framei->bias_g, m_acc[imuStartIdx] - Framei->bias_a, delta_t);
				if (imu_time_stamp[imuStartIdx + 1] >= time_end)
					break;
				imuStartIdx++;
			}

			Vec3 g_w;
			g_w << 0, 0, -G_norm;

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

			Mat915 J_imuj = Mat915::Zero();
			J_imuj.block(0, 0, 3, 3) = J_resP_p_j;
			J_imuj.block(3, 3, 3, 3) = J_resPhi_phi_j;
			J_imuj.block(6, 6, 3, 3) = J_resV_v_j;

			Mat99 Weight = Mat99::Zero();
			Weight.block(0, 0, 3, 3) = Cov.block(0, 0, 3, 3);
			Weight.block(3, 3, 3, 3) = Cov.block(6, 6, 3, 3);
			Weight.block(6, 6, 3, 3) = Cov.block(3, 3, 3, 3);

			Weight = Weight.diagonal().asDiagonal();
			Weight = imu_weight * imu_weight * Weight.inverse();

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
			//HM_change_half = HM_change_half * setting_margWeightFac_imu;
			//bM_change_half = bM_change_half * setting_margWeightFac_imu;

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
			Mat66 weight_bias = Mat66::Identity() * imu_weight * imu_weight * Cov_bias.inverse();
			HM_bias += (J_all_bias.transpose() * weight_bias * J_all_bias * setting_margWeightFac_imu);
			bM_bias += (J_all_bias.transpose() * weight_bias * r_all_bias * setting_margWeightFac_imu);
		}

		HM_change = HM_change * setting_margWeightFac_imu;
		bM_change = bM_change * setting_margWeightFac_imu;

		HM_change_half = HM_change_half * setting_margWeightFac_imu;
		bM_change_half = bM_change_half * setting_margWeightFac_imu;

		// 获取视觉部分得到的滑窗内优化量增量信息
		VecX StitchedDelta = getStitchedDeltaF();
		VecX delta_b = VecX::Zero(CPARS + 7 + nFrames * 17);

		// 将获取的视觉部分得到关键帧位姿增量赋值给待边缘化帧的前后两帧
		for (int fhIdx = fh->idx - 1; fhIdx < fh->idx + 1; ++fhIdx)
		{
			if (fhIdx < 0) continue;

			time_start = pic_time_stamp[frames[fhIdx]->data->shell->incoming_id];
			time_end = pic_time_stamp[frames[fhIdx + 1]->data->shell->incoming_id];
			dt = time_end - time_start;

			if (dt > 0.5) continue;

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

		bM_change -= HM_change * delta_b;
		bM_change_half -= HM_change_half * delta_b_half;

		double s_now = T_WD.scale(), di = 1;
		if (s_last > s_now) di = (s_last + 0.001) / s_now;
		else di = (s_now + 0.001) / s_last;
		s_last = s_now;

		if (di > d_now) d_now = di;
		if (d_now > d_min) d_now = d_min;

		LOG(INFO) << "s_now: " << s_now << " s_middle: " << s_middle << " d_now: " << d_now << " scale_l: " << T_WD_l.scale();
		
		if (di > d_half) d_half = di;

		if (d_half > d_min) d_half = d_min;
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
			if (d_half > d_min) d_half = d_min;

			marg_num_half = 0;
			T_WD_l_half = T_WD;
		}

		marg_num_half++;
		side_last = side;

		HM_imu_half += HM_change_half;
		bM_imu_half += bM_change_half;

		// 边缘化HM_imu_half中待去除的关键帧信息，将待边缘化的信息移动到矩阵的右下角
		{
			if ((int)fh->idx != (int)frames.size() - 1)
			{
				int io = fh->idx * 17 + CPARS + 7;	// index of frame to move to end
				int ntail = 17 * (nFrames - fh->idx - 1);
				assert((io + 17 + ntail) == nFrames * 17 + CPARS + 7);

				Vec17 bTmp = bM_imu_half.segment<17>(io);
				VecX tailTmp = bM_imu_half.tail(ntail);
				bM_imu_half.segment(io, ntail) = tailTmp;
				bM_imu_half.tail<17>() = bTmp;

				MatXX HtmpCol = HM_imu_half.block(0, io, odim, 17);
				MatXX rightColsTmp = HM_imu_half.rightCols(ntail);
				HM_imu_half.block(0, io, odim, ntail) = rightColsTmp;
				HM_imu_half.rightCols(17) = HtmpCol;

				MatXX HtmpRow = HM_imu_half.block(io, 0, 17, odim);
				MatXX botRowsTmp = HM_imu_half.bottomRows(ntail);
				HM_imu_half.block(io, 0, ntail, odim) = botRowsTmp;
				HM_imu_half.bottomRows(17) = HtmpRow;
			}

			VecX SVec = (HM_imu_half.diagonal().cwiseAbs() + VecX::Constant(HM_imu_half.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();

			MatXX HMScaled = SVecI.asDiagonal() * HM_imu_half * SVecI.asDiagonal();
			VecX bMScaled = SVecI.asDiagonal() * bM_imu_half;

			Mat1717 hpi = HMScaled.bottomRightCorner<17, 17>();
			hpi = 0.5f * (hpi + hpi);
			hpi = hpi.inverse();
			hpi = 0.5f * (hpi + hpi);
			if (!std::isfinite(hpi(0, 0))) hpi = Mat1717::Zero();

			MatXX bli = HMScaled.bottomLeftCorner(17, ndim).transpose() * hpi;
			HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17, ndim);
			bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<17>();

			HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
			bMScaled = SVec.asDiagonal() * bMScaled;

			HM_imu_half = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
			bM_imu_half = bMScaled.head(ndim);
		}

		// 边缘化惯导信息中的偏置信息
		// TODO：为什么惯导偏置信息要和IMU的其他信息分开边缘化
		{
			if ((int)fh->idx != (int)frames.size() - 1)
			{
				int io = fh->idx * 17 + CPARS + 7;	// index of frame to move to end
				int ntail = 17 * (nFrames - fh->idx - 1);
				assert((io + 17 + ntail) == nFrames * 17 + CPARS + 7);

				Vec17 bTmp = bM_bias.segment<17>(io);
				VecX tailTMP = bM_bias.tail(ntail);
				bM_bias.segment(io, ntail) = tailTMP;
				bM_bias.tail<17>() = bTmp;

				MatXX HtmpCol = HM_bias.block(0, io, odim, 17);
				MatXX rightColsTmp = HM_bias.rightCols(ntail);
				HM_bias.block(0, io, odim, ntail) = rightColsTmp;
				HM_bias.rightCols(17) = HtmpCol;

				MatXX HtmpRow = HM_bias.block(io, 0, 17, odim);
				MatXX botRowsTmp = HM_bias.bottomRows(ntail);
				HM_bias.block(io, 0, ntail, odim) = botRowsTmp;
				HM_bias.bottomRows(17) = HtmpRow;
			}
			VecX SVec = (HM_bias.diagonal().cwiseAbs() + VecX::Constant(HM_bias.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();

			MatXX HMScaled = SVecI.asDiagonal() * HM_bias * SVecI.asDiagonal();
			VecX bMScaled = SVecI.asDiagonal() * bM_bias;

			Mat1717 hpi = HMScaled.bottomRightCorner<17, 17>();
			hpi = 0.5f*(hpi + hpi);
			hpi = hpi.inverse();
			hpi = 0.5f*(hpi + hpi);
			if (!std::isfinite(hpi(0, 0))) hpi = Mat1717::Zero();

			MatXX bli = HMScaled.bottomLeftCorner(17, ndim).transpose() * hpi;
			HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17, ndim);
			bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<17>();

			HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
			bMScaled = SVec.asDiagonal() * bMScaled;

			HM_bias = 0.5*(HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
			bM_bias = bMScaled.head(ndim);
		}

		// TODO：是否意味着尺度信息已经收敛了
		if (marg_num > 25) use_Dmargin = false;

		if ((s_now > s_middle*d_now || s_now < s_middle / d_now) && use_Dmargin) 
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
			if (d_half > d_min) d_half = d_min;
			marg_num_half = 0;
			T_WD_l = T_WD_l_half;
			state_twd = Sim3(T_WD_l.inverse()*T_WD).log();
		}
		else
		{
			HM_imu += HM_change;
			bM_imu += bM_change;
			marg_num++;

			if ((int)fh->idx != (int)frames.size() - 1)
			{
				int io = fh->idx * 17 + CPARS + 7;
				int ntail = 17 * (nFrames - fh->idx - 1);
				assert((io + 17 + ntail) == nFrames * 17 + CPARS + 7);

				Vec17 bTmp = bM_imu.segment<17>(io);
				VecX tailTMP = bM_imu.tail(ntail);
				bM_imu.segment(io, ntail) = tailTMP;
				bM_imu.tail<17>() = bTmp;

				MatXX HtmpCol = HM_imu.block(0, io, odim, 17);
				MatXX rightColsTmp = HM_imu.rightCols(ntail);
				HM_imu.block(0, io, odim, ntail) = rightColsTmp;
				HM_imu.rightCols(17) = HtmpCol;

				MatXX HtmpRow = HM_imu.block(io, 0, 17, odim);
				MatXX botRowsTmp = HM_imu.bottomRows(ntail);
				HM_imu.block(io, 0, ntail, odim) = botRowsTmp;
				HM_imu.bottomRows(17) = HtmpRow;
			}
			VecX SVec = (HM_imu.diagonal().cwiseAbs() + VecX::Constant(HM_imu.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();

			MatXX HMScaled = SVecI.asDiagonal() * HM_imu * SVecI.asDiagonal();
			VecX bMScaled = SVecI.asDiagonal() * bM_imu;

			Mat1717 hpi = HMScaled.bottomRightCorner<17, 17>();
			hpi = 0.5f*(hpi + hpi);
			hpi = hpi.inverse();
			hpi = 0.5f*(hpi + hpi);
			if (!std::isfinite(hpi(0, 0))) hpi = Mat1717::Zero();

			MatXX bli = HMScaled.bottomLeftCorner(17, ndim).transpose() * hpi;
			HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17, ndim);
			bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<17>();

			HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
			bMScaled = SVec.asDiagonal() * bMScaled;

			HM_imu = 0.5*(HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
			bM_imu = bMScaled.head(ndim);
		}
	}

	void EnergyFunctional::marginalizeFrame(EFFrame* fh)
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		assert((int)fh->points.size() == 0);

		if (imu_use_flag)
			marginalizeFrame_imu(fh);

		int ndim = nFrames * 8 + CPARS - 8;// new dimension
		int odim = nFrames * 8 + CPARS;// old dimension

		// 若需要marg的帧不是关键帧中最后一帧则需要将这帧数据移到矩阵最后
		if ((int)fh->idx != (int)frames.size() - 1)
		{
			int io = fh->idx * 8 + CPARS;
			int ntail = 8 * (nFrames - fh->idx - 1);
			assert((io + 8 + ntail) == nFrames * 8 + CPARS);

			Vec8 bTmp = bM.segment<8>(io);
			VecX tailTmp = bM.tail(ntail);
			bM.segment(io, ntail) = tailTmp;
			bM.tail<8>() = bTmp;

			MatXX HtmpCol = HM.block(0, io, odim, 8);
			MatXX rightColsTmp = HM.rightCols(ntail);
			HM.block(0, io, odim, ntail) = rightColsTmp;
			HM.rightCols(8) = HtmpCol;

			MatXX HtmpRow = HM.block(io, 0, 8, odim);
			MatXX botRowsTmp = HM.bottomRows(ntail);
			HM.block(io, 0, ntail, odim) = botRowsTmp;
			HM.bottomRows(8) = HtmpRow;
		}

		// marginalize. First add prior here, instead of to active.
		HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
		bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

		VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
		VecX SVecI = SVec.cwiseInverse();

		MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
		VecX bMScaled = SVecI.asDiagonal() * bM;

		Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
		hpi = 0.5f*(hpi + hpi);
		hpi = hpi.inverse();
		hpi = 0.5f*(hpi + hpi);
		if (!std::isfinite(hpi(0, 0))) hpi = Mat88::Zero();

		// 舒尔补求解边缘化后的Hessian以及b
		MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
		HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
		bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

		HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
		bMScaled = SVec.asDiagonal() * bMScaled;

		HM = 0.5*(HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
		bM = bMScaled.head(ndim);
		
		// Hessian矩阵以及b矩阵中的信息去除后，去掉对应的关键帧信息
		for (unsigned int i = fh->idx; i + 1 < frames.size(); i++)
		{
			frames[i] = frames[i + 1];
			frames[i]->idx = i;
		}

		nFrames--;
		frames.pop_back();
		fh->data->efFrame = 0;

		EFIndicesValid = false;
		EFAdjointsValid = false;
		EFDeltaValid = false;

		makeIDX();
		delete fh; fh = NULL;
	}

	/// <summary>
	/// 边缘化关键帧时，关键帧中包含很多路标点信息，若直接丢弃则优化问题中
	/// 会损失很多信息，此时的做法是将有效的路标点收集起来进行边缘化求解Hessian
	/// </summary>
	void EnergyFunctional::marginalizePointsF()
	{
		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		allPointsToMarg.clear();

		// 收集待边缘化帧中有效路标点
		for (EFFrame* f : frames)
		{
			for (int i = 0; i < (int)f->points.size(); ++i)
			{
				EFPoint* p = f->points[i];
				if (p->stateFlag == EFPointStatus::PS_MARGINALIZE)
				{
					p->priorF *= setting_idepthFixPriorMargFac;
					for (EFResidual* r : p->residualsAll)
						if (r->isActive())
							if (!r->data->stereoResidualFlag)
								connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;

					int ngoodres = 0;
					for (EFResidual* r : p->residualsAll)
						if (r->isActive() && !r->data->stereoResidualFlag) ngoodres++;
					if (ngoodres > 0)
						allPointsToMarg.emplace_back(p);
					else
						removePoint(p);
				}
			}
		}

		// 使用有效路标点边缘化计算Heesian以及b
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

		HM += setting_margWeightFac * H;
		bM += setting_margWeightFac * b;

		if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
			orthogonalize(&bM, &HM);

		EFIndicesValid = false;
		makeIDX();
	}

	void EnergyFunctional::dropPointsF()
	{
		for (EFFrame* f : frames)
		{
			for (int i = 0; i < (int)f->points.size(); i++)
			{
				EFPoint* p = f->points[i];
				if (p->stateFlag == EFPointStatus::PS_DROP)
				{
					removePoint(p);
					i--;
				}
			}
		}

		EFIndicesValid = false;
		makeIDX();
	}

	void EnergyFunctional::removePoint(EFPoint* p)
	{
		// 删除由该点构造的所有残差
		for (EFResidual* r : p->residualsAll)
			dropResidual(r);

		// 在该点的host帧中删除该点
		EFFrame* h = p->host;
		h->points[p->idxInPoints] = h->points.back();
		h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
		h->points.pop_back();

		nPoints--;
		p->data->efPoint = 0;

		EFIndicesValid = false;

		delete p; p = NULL;
	}

	void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
	{
		//	VecX eigenvaluesPre = H.eigenvalues().real();
		//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
		//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

		// decide to which nullspaces to orthogonalize.
		std::vector<VecX> ns;
		ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
		ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
		//	if(setting_affineOptModeA <= 0)
		//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
		//	if(setting_affineOptModeB <= 0)
		//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

		// make Nullspaces matrix
		MatXX N(ns[0].rows(), ns.size());
		for (unsigned int i = 0; i < ns.size(); i++)
			N.col(i) = ns[i].normalized();

		// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
		Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX SNN = svdNN.singularValues();
		double minSv = 1e10, maxSv = 0;
		for (int i = 0; i < SNN.size(); i++)
		{
			if (SNN[i] < minSv) minSv = SNN[i];
			if (SNN[i] > maxSv) maxSv = SNN[i];
		}
		for (int i = 0; i < SNN.size(); i++)
		{
			if (SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0;
		}

		MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
		MatXX NNpiT = N * Npi.transpose(); 	// [dim] x [dim].
		MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

		if (b != 0) *b -= NNpiTS * *b;
		if (H != 0) *H -= NNpiTS * *H * NNpiTS;

		//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

		//	VecX eigenvaluesPost = H.eigenvalues().real();
		//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
		//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
	}

	// iteration：优化迭代次数
	// lambda：优化迭代中的控制迭代收敛速度的因子
	// HCalib：相机内参数信息
	void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
	{
		if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
		if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

		assert(EFDeltaValid);
		assert(EFAdjointsValid);
		assert(EFIndicesValid);

		MatXX HL_top, HA_top, H_sc, H_imu, HFinal_top;
		VecX  bL_top, bA_top, b_sc, b_imu, bFinal_top;

		calcIMUHessian(H_imu, b_imu);

		//std::cout << "Himu: " << std::endl << H_imu << std::endl;
		//std::cout << "bimu: " << std::endl << b_imu << std::endl;

		accumulateAF_MT(HA_top, bA_top, multiThreading);
		accumulateLF_MT(HL_top, bL_top, multiThreading);
		accumulateSCF_MT(H_sc, b_sc, multiThreading);

		VecX StitchedDeltaVisual = getStitchedDeltaF();
		VecX StitchedDeltaIMU = VecX::Zero(CPARS + 7 + nFrames * 17);
		// 	StitchedDelta2.block(0,0,CPARS,1) = StitchedDelta.block(0,0,CPARS,1);
		for (int idx = 0; idx < nFrames; ++idx)
		{
			if (frames[idx]->m_flag)
				StitchedDeltaIMU.block(CPARS + 7 + 17 * idx, 0, 6, 1) = StitchedDeltaVisual.block(CPARS + 8 * idx, 0, 6, 1);
		}
		StitchedDeltaIMU.block(CPARS, 0, 7, 1) = state_twd;

		VecX bM_top = (bM + HM * StitchedDeltaVisual);
		VecX bM_top_imu = (bM_imu + HM_imu * StitchedDeltaIMU);

		if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
		{
			bool haveFirstFrame = false;
			for (EFFrame* f : frames)
				if (f->frameID == 0) haveFirstFrame = true;

			MatXX HT_act = HL_top + HA_top - H_sc;
			VecX bT_act = bL_top + bA_top - b_sc;

			if (!haveFirstFrame)
				orthogonalize(&bT_act, &HT_act);

			HFinal_top = HT_act + HM;
			bFinal_top = bT_act + bM_top;

			lastHS = HFinal_top;
			lastbS = bFinal_top;

			for (int idx = 0; idx < 8 * nFrames + CPARS; idx++)
				HFinal_top(idx, idx) *= (1 + lambda);
		}
		else
		{
			HFinal_top = HL_top + HM + HA_top;
			bFinal_top = bL_top + bM_top + bA_top - b_sc;

			lastHS = HFinal_top - H_sc;
			lastbS = bFinal_top;

			for (int idx = 0; idx < 8 * nFrames + CPARS; idx++)
				HFinal_top(idx, idx) *= (1 + lambda);
			HFinal_top -= H_sc * (1.0f / (1 + lambda));
		}

		H_imu(6, 6) += setting_initialScaleHessian;
		H_imu.block(3, 3, 3, 3) += setting_initialIMUHessian * Mat33::Identity();
		
		for (int idx = 0; idx < nFrames; ++idx)
		{
			H_imu.block(7 + 15 * idx + 9, 7 + 15 * idx + 9, 3, 3) += setting_initialbgHessian * Mat33::Identity();
			H_imu.block(7 + 15 * idx + 12, 7 + 15 * idx + 12, 3, 3) += setting_initialbaHessian * Mat33::Identity();
		}
		for (int idx = 0; idx < 7 + 15 * nFrames; ++idx) H_imu(idx, idx) *= (1 + lambda);

		// 构造结合惯导信息以及视觉信息的Hessian以及b且Hessian矩阵的排列顺序如下
		// 相机内参、世界系与DSO系的变化、帧姿态、光度参数、速度、陀螺仪偏置、加速度计偏置
		MatXX HFinal_top2 = MatXX::Zero(CPARS + 7 + 17 * nFrames, CPARS + 7 + 17 * nFrames);
		VecX bFinal_top2 = VecX::Zero(CPARS + 7 + 17 * nFrames);
		HFinal_top2.block(0, 0, CPARS, CPARS) = HFinal_top.block(0, 0, CPARS, CPARS);
		HFinal_top2.block(CPARS, CPARS, 7, 7) = H_imu.block(0, 0, 7, 7);
		bFinal_top2.block(0, 0, CPARS, 1) = bFinal_top.block(0, 0, CPARS, 1);
		bFinal_top2.block(CPARS, 0, 7, 1) = b_imu.block(0, 0, 7, 1);

		for (int idx = 0; idx < nFrames; ++idx)
		{
			// 相机内参与其它关键帧位姿和光度参数的关系
			HFinal_top2.block(0, CPARS + 7 + idx * 17, CPARS, 8) += HFinal_top.block(0, CPARS + idx * 8, CPARS, 8);
			HFinal_top2.block(CPARS + 7 + idx * 17, 0, 8, CPARS) += HFinal_top.block(CPARS + idx * 8, 0, 8, CPARS);

			// Twd与IMU的P、V、Q、Bias之间的关系
			HFinal_top2.block(CPARS, CPARS + 7 + idx * 17, 7, 6) += H_imu.block(0, 7 + idx * 15, 7, 6);
			HFinal_top2.block(CPARS + 7 + idx * 17, CPARS, 6, 7) += H_imu.block(7 + idx * 15, 0, 6, 7);
			HFinal_top2.block(CPARS, CPARS + 7 + idx * 17 + 8, 7, 9) += H_imu.block(0, 7 + idx * 15 + 6, 7, 9);
			HFinal_top2.block(CPARS + 7 + idx * 17 + 8, CPARS, 9, 7) += H_imu.block(7 + idx * 15 + 6, 0, 9, 7);

			// 相机位姿、光度参数
			HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17, 8, 8) += HFinal_top.block(CPARS + idx * 8, CPARS + idx * 8, 8, 8);
			HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17, 6, 6) += H_imu.block(7 + idx * 15, 7 + idx * 15, 6, 6);

			// 速度、加速度偏置、陀螺仪偏置
			HFinal_top2.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + idx * 17 + 8, 9, 9) += H_imu.block(7 + idx * 15 + 6, 7 + idx * 15 + 6, 9, 9);
			HFinal_top2.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + idx * 17, 9, 6) += H_imu.block(7 + idx * 15 + 6, 7 + idx * 15, 9, 6);
			HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + idx * 17 + 8, 6, 9) += H_imu.block(7 + idx * 15, 7 + idx * 15 + 6, 6, 9);

			for (int j = idx + 1; j < nFrames; ++j) 
			{
				//pose a b
				HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17, 8, 8) += HFinal_top.block(CPARS + idx * 8, CPARS + j * 8, 8, 8);
				HFinal_top2.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17, 8, 8) += HFinal_top.block(CPARS + j * 8, CPARS + idx * 8, 8, 8);
				//pose
				HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17, 6, 6) += H_imu.block(7 + idx * 15, 7 + j * 15, 6, 6);
				HFinal_top2.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17, 6, 6) += H_imu.block(7 + j * 15, 7 + idx * 15, 6, 6);
				//v bg ba
				HFinal_top2.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + j * 17 + 8, 9, 9) += H_imu.block(7 + idx * 15 + 6, 7 + j * 15 + 6, 9, 9);
				HFinal_top2.block(CPARS + 7 + j * 17 + 8, CPARS + 7 + idx * 17 + 8, 9, 9) += H_imu.block(7 + j * 15 + 6, 7 + idx * 15 + 6, 9, 9);
				//v bg ba,pose
				HFinal_top2.block(CPARS + 7 + idx * 17 + 8, CPARS + 7 + j * 17, 9, 6) += H_imu.block(7 + idx * 15 + 6, 7 + j * 15, 9, 6);
				HFinal_top2.block(CPARS + 7 + j * 17, CPARS + 7 + idx * 17 + 8, 6, 9) += H_imu.block(7 + j * 15, 7 + idx * 15 + 6, 6, 9);
				//pose,v bg ba
				HFinal_top2.block(CPARS + 7 + idx * 17, CPARS + 7 + j * 17 + 8, 6, 9) += H_imu.block(7 + idx * 15, 7 + j * 15 + 6, 6, 9);
				HFinal_top2.block(CPARS + 7 + j * 17 + 8, CPARS + 7 + idx * 17, 9, 6) += H_imu.block(7 + j * 15 + 6, 7 + idx * 15, 9, 6);
			}

			bFinal_top2.block(CPARS + 7 + 17 * idx, 0, 8, 1) += bFinal_top.block(CPARS + 8 * idx, 0, 8, 1);
			bFinal_top2.block(CPARS + 7 + 17 * idx, 0, 6, 1) += b_imu.block(7 + 15 * idx, 0, 6, 1);
			bFinal_top2.block(CPARS + 7 + 17 * idx + 8, 0, 9, 1) += b_imu.block(7 + 15 * idx + 6, 0, 9, 1);
		}

		HFinal_top2 += (HM_imu + HM_bias);
		// 	bFinal_top2 += (bM_imu + bM_bias);
		bFinal_top2 += (bM_top_imu + bM_bias);
		VecX x = VecX::Zero(CPARS + 8 * nFrames);
		VecX x2 = VecX::Zero(CPARS + 7 + 17 * nFrames);
		VecX x3 = VecX::Zero(CPARS + 7 + 17 * nFrames);

		if (setting_solverMode & SOLVER_SVD)
		{
			VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
			MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
			VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
			Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

			VecX S = svd.singularValues();
			double minSv = 1e10, maxSv = 0;
			for (int i = 0; i < S.size(); i++)
			{
				if (S[i] < minSv) minSv = S[i];
				if (S[i] > maxSv) maxSv = S[i];
			}

			VecX Ub = svd.matrixU().transpose()*bFinalScaled;
			int setZero = 0;
			for (int i = 0; i < Ub.size(); i++)
			{
				if (S[i] < setting_solverModeDelta*maxSv)
				{
					Ub[i] = 0; setZero++;
				}

				if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7))
				{
					Ub[i] = 0; setZero++;
				}

				else Ub[i] /= S[i];
			}
			x = SVecI.asDiagonal() * svd.matrixV() * Ub;
		}
		else
		{
			if (!imu_use_flag)
			{
				VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
				x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
			}
			else
			{
				//std::cout << "Himu: " << H_imu << std::endl;
				//std::cout << "HFinal_top: " << HFinal_top << std::endl;
				//std::cout << "HFinal_top2: " << HFinal_top2 << std::endl;
				//std::cout << "bFinal_top2: " << bFinal_top2 << std::endl;

				VecX SVecI = (HFinal_top2.diagonal() + VecX::Constant(HFinal_top2.cols(), 10)).cwiseSqrt().cwiseInverse();
				MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top2 * SVecI.asDiagonal();
				x2 = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top2);//  SVec.asDiagonal() * svd.matrixV() * Ub;

				//std::cout << "SVecI: " << SVecI << std::endl;
				//std::cout << "HFinal_top: " << HFinal_top2 << std::endl;
				//std::cout << "bFinal_top: " << bFinal_top2 << std::endl;

				x.block(0, 0, CPARS, 1) = x2.block(0, 0, CPARS, 1);
				for (int i = 0; i < nFrames; ++i)
				{
					x.block(CPARS + i * 8, 0, 8, 1) = x2.block(CPARS + 7 + 17 * i, 0, 8, 1);
					frames[i]->data->step_imu = -x2.block(CPARS + 7 + 17 * i + 8, 0, 9, 1);
				}
				step_twd = -x2.block(CPARS, 0, 7, 1);
			}
		}

		if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
		{
			VecX xOld = x;
			orthogonalize(&x, 0);
		}

		lastX = x;
		currentLambda = lambda;
		resubstituteF_MT(x, HCalib, multiThreading);
		currentLambda = 0;
	}

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
	/// 获取视觉部分优化得到的相机内参、相机姿态以及相机光度系数delta
	/// </summary>
	/// <returns></returns>
	VecX EnergyFunctional::getStitchedDeltaF() const
	{
		VecX d = VecX(CPARS + nFrames * 8);
		d.head<CPARS>() = cDeltaF.cast<double>();
		for (int h = 0; h < nFrames; ++h)
			d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
		return d;
	}
}