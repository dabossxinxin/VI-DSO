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

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"

namespace dso
{
	/// <summary>
	/// 绘制关键帧特征的跟踪以及观测残差情况：具体绘制内容为；
	/// 对于每一个滑窗中的关键帧，各自管理一些特征点，将特征点绘制在主帧中，并按照特征深度绘制热力图；
	/// 对于每一个特征，管理一系列观测残差，将特征的每一个观测投影到目标帧中，并绘制在目标帧中的投影；
	/// 综上所述，这个函数的作用为绘制滑窗中所有特征在主帧上的像素坐标以及在目标帧上的投影；
	/// </summary>
	void FullSystem::debugPlotTracking()
	{
		if (setting_disableAllDisplay) return;
		if (!setting_render_plotTrackingFull) return;
		int wh = hG[0] * wG[0];

		// 将滑窗中关键帧特征相互投影
		int idx = 0;
		for (FrameHessian* f : frameHessians)
		{
			std::vector<MinimalImageB3*> images;

			for (FrameHessian* f2 : frameHessians)
				if (f2->debugImage == NULL)
					f2->debugImage = new MinimalImageB3(wG[0], hG[0]);

			// 绘制关键帧的图像信息
			for (FrameHessian* f2 : frameHessians)
			{
				MinimalImageB3* debugImage = f2->debugImage;
				images.emplace_back(debugImage);

				Eigen::Vector3f* fd = f2->dI;

				Vec2 affL = AffLight::fromToVecExposure(f2->ab_exposure, f->ab_exposure, f2->aff_g2l(), f->aff_g2l());

				for (int it = 0; it < wh; ++it)
				{
					// BRIGHTNESS TRANSFER
					float colL = affL[0] * fd[it][0] + affL[1];
					if (colL < 0) colL = 0;
					if (colL > 255) colL = 255;
					debugImage->at(it) = Vec3b(colL, colL, colL);
				}
			}

			// 绘制关键帧的逆深度信息
			for (PointHessian* ph : f->pointHessians)
			{
				if (ph->status == PointHessian::ACTIVE || ph->status == PointHessian::MARGINALIZED)
				{
					// 将主帧特征投影到目标帧中并绘制
					for (PointFrameResidual* r : ph->residuals)
						r->debugPlot();

					// 将主帧特征在主帧中根据特征逆深度绘制热力图
					f->debugImage->setPixel9(ph->u + 0.5, ph->v + 0.5, makeRainbow3B(ph->idepth_scaled));
				}
			}

			char buf[100];
			snprintf(buf, 100, "IMG %d", idx);
			IOWrap::displayImageStitch(buf, images);
			idx++;
		}

		//IOWrap::waitKey(0);
	}

	void FullSystem::debugPlot(std::string name)
	{
		if (setting_disableAllDisplay) return;
		if (!setting_render_renderWindowFrames) return;
		std::vector<MinimalImageB3*> images;

		float minID = 0, maxID = 0;
		if ((int)(freeDebugParam5 + 0.5f) == 7 || (setting_debugSaveImages && false))
		{
			std::vector<float> allID;
			for (unsigned int f = 0; f < frameHessians.size(); f++)
			{
				for (PointHessian* ph : frameHessians[f]->pointHessians)
					if (ph != 0) allID.push_back(ph->idepth_scaled);

				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
					if (ph != 0) allID.push_back(ph->idepth_scaled);

				for (PointHessian* ph : frameHessians[f]->pointHessiansOut)
					if (ph != 0) allID.push_back(ph->idepth_scaled);
			}
			std::sort(allID.begin(), allID.end());
			int n = allID.size() - 1;
			minID = allID[(int)(n * 0.05)];
			maxID = allID[(int)(n * 0.95)];

			// slowly adapt: change by maximum 10% of old span.
			float maxChange = 0.1 * (maxIdJetVisDebug - minIdJetVisDebug);
			if (maxIdJetVisDebug < 0 || minIdJetVisDebug < 0) maxChange = 1e5;

			if (minID < minIdJetVisDebug - maxChange)
				minID = minIdJetVisDebug - maxChange;
			if (minID > minIdJetVisDebug + maxChange)
				minID = minIdJetVisDebug + maxChange;


			if (maxID < maxIdJetVisDebug - maxChange)
				maxID = maxIdJetVisDebug - maxChange;
			if (maxID > maxIdJetVisDebug + maxChange)
				maxID = maxIdJetVisDebug + maxChange;

			maxIdJetVisDebug = maxID;
			minIdJetVisDebug = minID;

		}

		int wh = hG[0] * wG[0];
		for (unsigned int f = 0; f < frameHessians.size(); f++)
		{
			MinimalImageB3* img = new MinimalImageB3(wG[0], hG[0]);
			images.push_back(img);
			//float* fd = frameHessians[f]->I;
			Eigen::Vector3f* fd = frameHessians[f]->dI;

			for (int i = 0; i < wh; i++)
			{
				int c = fd[i][0] * 0.9f;
				if (c > 255) c = 255;
				img->at(i) = Vec3b(c, c, c);
			}

			if ((int)(freeDebugParam5 + 0.5f) == 0)
			{
				for (PointHessian* ph : frameHessians[f]->pointHessians)
				{
					if (ph == 0) continue;

					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeRainbow3B(ph->idepth_scaled));
				}
				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
				{
					if (ph == 0) continue;
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeRainbow3B(ph->idepth_scaled));
				}
				for (PointHessian* ph : frameHessians[f]->pointHessiansOut)
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
			}
			else if ((int)(freeDebugParam5 + 0.5f) == 1)
			{
				for (PointHessian* ph : frameHessians[f]->pointHessians)
				{
					if (ph == 0) continue;
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeRainbow3B(ph->idepth_scaled));
				}

				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));

				for (PointHessian* ph : frameHessians[f]->pointHessiansOut)
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
			}
			else if ((int)(freeDebugParam5 + 0.5f) == 2)
			{

			}
			else if ((int)(freeDebugParam5 + 0.5f) == 3)
			{
				for (ImmaturePoint* ph : frameHessians[f]->immaturePoints)
				{
					if (ph == 0) continue;
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD ||
						ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ||
						ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
					{
						if (!std::isfinite(ph->idepth_max))
							img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
						else
						{
							img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeRainbow3B((ph->idepth_min + ph->idepth_max) * 0.5f));
						}
					}
				}
			}
			else if ((int)(freeDebugParam5 + 0.5f) == 4)
			{
				for (ImmaturePoint* ph : frameHessians[f]->immaturePoints)
				{
					if (ph == 0) continue;

					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 0));
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 0));
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 255, 255));
					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
				}
			}
			else if ((int)(freeDebugParam5 + 0.5f) == 5)
			{
				for (ImmaturePoint* ph : frameHessians[f]->immaturePoints)
				{
					if (ph == 0) continue;

					if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) continue;
					float d = freeDebugParam1 * (sqrtf(ph->quality) - 1);
					if (d < 0) d = 0;
					if (d > 1) d = 1;
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, d * 255, (1 - d) * 255));
				}

			}
			else if ((int)(freeDebugParam5 + 0.5f) == 6)
			{
				for (PointHessian* ph : frameHessians[f]->pointHessians)
				{
					if (ph == 0) continue;
					if (ph->my_type == 0)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 255));
					if (ph->my_type == 1)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
					if (ph->my_type == 2)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
					if (ph->my_type == 3)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 255));
				}
				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
				{
					if (ph == 0) continue;
					if (ph->my_type == 0)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 255));
					if (ph->my_type == 1)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(255, 0, 0));
					if (ph->my_type == 2)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 255));
					if (ph->my_type == 3)
						img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 255, 255));
				}

			}
			if ((int)(freeDebugParam5 + 0.5f) == 7)
			{
				for (PointHessian* ph : frameHessians[f]->pointHessians)
				{
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))));
				}
				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
				{
					if (ph == 0) continue;
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
				}
			}
		}
		IOWrap::displayImageStitch(name.c_str(), images);
		IOWrap::waitKey(5);

		for (unsigned int i = 0; i < images.size(); i++)
			delete images[i];

		if ((setting_debugSaveImages && false))
		{
			for (unsigned int f = 0; f < frameHessians.size(); f++)
			{
				MinimalImageB3* img = new MinimalImageB3(wG[0], hG[0]);
				Eigen::Vector3f* fd = frameHessians[f]->dI;

				for (int i = 0; i < wh; i++)
				{
					int c = fd[i][0] * 0.9f;
					if (c > 255) c = 255;
					img->at(i) = Vec3b(c, c, c);
				}

				for (PointHessian* ph : frameHessians[f]->pointHessians)
				{
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))));
				}
				for (PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
				{
					if (ph == 0) continue;
					img->setPixelCirc(ph->u + 0.5f, ph->v + 0.5f, Vec3b(0, 0, 0));
				}

				char buf[1000];
				snprintf(buf, 1000, "images_out/kf_%05d_%05d_%02d.png",
					frameHessians.back()->shell->id, frameHessians.back()->frameID, f);
				IOWrap::writeImage(buf, img);

				delete img;
			}
		}
	}
}