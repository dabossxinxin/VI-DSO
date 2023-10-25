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
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{
	PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
	{
		instanceCounter++;
		host = rawPoint->host;
		hasDepthPrior = false;

		idepth_hessian = 0;
		maxRelBaseline = 0;
		numGoodResiduals = 0;

		// set static values & initialization.
		u = rawPoint->u;
		v = rawPoint->v;
		assert(std::isfinite(rawPoint->idepth_max));
		//idepth_init = rawPoint->idepth_GT;

		my_type = rawPoint->my_type;

		setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
		setPointStatus(PointHessian::INACTIVE);

		int n = patternNum;
		memcpy(color, rawPoint->color, sizeof(float)*n);
		memcpy(weights, rawPoint->weights, sizeof(float)*n);
		energyTH = rawPoint->energyTH;

		efPoint = 0;
	}

	void PointHessian::release()
	{
		for (unsigned int i = 0; i < residuals.size(); i++)
			delete residuals[i];
		residuals.clear();
	}

	/// <summary>
	/// 求解当前帧位姿参数以及光度参数的零空间
	/// </summary>
	/// <param name="state_zero"></param>
	void FrameHessian::setStateZero(const Vec10& state_zero)
	{
		assert(state_zero.head<6>().squaredNorm() < 1e-20);
		this->state_zero = state_zero;
		Vec6 eps;

		// 计算位姿零空间
		for (int it = 0; it < 6; ++it)
		{
			eps.setZero();
			eps[it] = 1e-3;

			SE3 EepsP = SE3::exp(eps);
			SE3 EepsM = SE3::exp(-eps);
			SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
			SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
			nullspaces_pose.col(it) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
		}

		// 计算尺度零空间
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
		w2c_leftEps_P_x0.translation() *= 1.001;
		w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
		w2c_leftEps_M_x0.translation() /= 1.001;
		w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
		nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

		// 计算光度参数零空间                                
		nullspaces_affine.setZero();
		nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
		assert(ab_exposure > 0);
		nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
	};

	/// <summary>
	/// 析构帧内内存空间
	/// </summary>
	void FrameHessian::release()
	{
		for (unsigned int it = 0; it < pointHessians.size(); ++it) delete pointHessians[it];
		for (unsigned int it = 0; it < pointHessiansMarginalized.size(); ++it) delete pointHessiansMarginalized[it];
		for (unsigned int it = 0; it < pointHessiansOut.size(); ++it) delete pointHessiansOut[it];
		for (unsigned int it = 0; it < immaturePoints.size(); ++it) delete immaturePoints[it];

		pointHessians.clear();
		pointHessiansMarginalized.clear();
		pointHessiansOut.clear();
		immaturePoints.clear();
	}

	/// <summary>
	/// 设置当前帧的灰度图像数据以及梯度数据
	/// </summary>
	/// <param name="color">输入灰度通道图像数据</param>
	/// <param name="HCalib">输入相机的标定参数</param>
	void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
	{
		// 为当前帧分配金字塔图像内存以及梯度内存
		for (int it = 0; it < pyrLevelsUsed; ++it)
		{
			dIp[it] = new Eigen::Vector3f[wG[it] * hG[it]];
			absSquaredGrad[it] = new float[wG[it] * hG[it]];
		}
		dI = dIp[0];

		// 设置第零层金字塔的图像数据
		int pixelNum = wG[0] * hG[0];
		for (int it = 0; it < pixelNum; ++it)
			dI[it][0] = color[it];

		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			int wl = wG[lvl], hl = hG[lvl];
			Eigen::Vector3f* dI_l = dIp[lvl];

			// 设置第lvl层金字塔的图像数据
			float* dabs_l = absSquaredGrad[lvl];
			if (lvl > 0)
			{
				int lvlm1 = lvl - 1;
				int wlm1 = wG[lvlm1];
				Eigen::Vector3f* dI_lm = dIp[lvlm1];

				// 使用像素均值作为采样图像对应像素的像素值
				for (int y = 0; y < hl; ++y)
				{
					for (int x = 0; x < wl; ++x)
					{
						dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
							dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
							dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
							dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
					}
				}
			}

			// 计算第lvl层金字塔图像的梯度
			for (int idx = wl; idx < wl * (hl - 1); ++idx)
			{
				float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
				float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

				if (!std::isfinite(dx)) dx = 0;
				if (!std::isfinite(dy)) dy = 0;

				dI_l[idx][1] = dx;
				dI_l[idx][2] = dy;

				dabs_l[idx] = dx * dx + dy * dy;

				if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
				{
					float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
					dabs_l[idx] *= gw * gw;	// convert to gradient of original color space (before removing response).
				}
			}
		}
	}

	void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib)
	{
		this->host = host;
		this->target = target;

		// 线性化点处host->target的位姿变换
		SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
		PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
		PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

		SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
		PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
		PRE_tTll = (leftToLeft.translation()).cast<float>();
		distanceLL = leftToLeft.translation().norm();

		Mat33f K = Mat33f::Zero();
		K(0, 0) = HCalib->fxl();
		K(1, 1) = HCalib->fyl();
		K(0, 2) = HCalib->cxl();
		K(1, 2) = HCalib->cyl();
		K(2, 2) = 1;
		PRE_KRKiTll = K * PRE_RTll * K.inverse();
		PRE_RKiTll = PRE_RTll * K.inverse();
		PRE_KtTll = K * PRE_tTll;

		PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
		PRE_b0_mode = host->aff_g2l_0().b;
	}
}