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
#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/nanoflann.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
	CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3())
	{
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			points[lvl] = nullptr;
			numPoints[lvl] = 0;
		}

		int wh = ww * hh;
		JbBuffer = new Vec10f[wh];
		JbBuffer_new = new Vec10f[wh];

		frameID = -1;
		printDebug = false;

		wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_TRANS;
		wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_ROT;
		wM.diagonal()[6] = SCALE_A;
		wM.diagonal()[7] = SCALE_B;
	}

	CoarseInitializer::~CoarseInitializer()
	{
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
			SAFE_DELETE(points[lvl], true);

		SAFE_DELETE(JbBuffer, true);
		SAFE_DELETE(JbBuffer_new, true);
	}

	/// <summary>
	/// 初始化中跟踪最新帧
	/// </summary>
	/// <param name="newFrame">最新帧数据</param>
	/// <param name="wraps">显示模块handle</param>
	/// <returns></returns>
	bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*>& wraps)
	{
		newFrame = newFrameHessian;

		for (IOWrap::Output3DWrapper* ow : wraps)
			ow->pushLiveFrame(newFrameHessian);

		// 初始化每层金字塔中的迭代次数
		int maxIterations[] = { 5,5,10,30,50 };

		// 设置正则项相关参数
		alphaK = 2.5 * 2.5;	//*freeDebugParam1*freeDebugParam1;
		alphaW = 150 * 150;	//*freeDebugParam2*freeDebugParam2;
		regWeight = 0.8;	//*freeDebugParam4;
		couplingWeight = 1;	//*freeDebugParam5;

		// 初始化handle中需要找到位移较大的帧进行初始化，因此若未找到，
		// 需要重置初始化handle中的位姿及特征
		if (!snapped)
		{
			thisToNext.translation().setZero();
			for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
			{
				int npts = numPoints[lvl];
				Pnt* ptsl = points[lvl];
				for (int it = 0; it < npts; ++it)
				{
					ptsl[it].iR = 1;
					ptsl[it].idepth_new = 1;
					ptsl[it].lastHessian = 0;
				}
			}
		}

		Vec3f latestRes = Vec3f::Zero();
		SE3 refToNew_current = thisToNext;
		AffLight refToNew_aff_current = thisToNext_aff;

		if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
			refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0);

		// 从金字塔顶层向金字塔底层逐层优化各项参数
		for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
		{
			// 将上层优化得到的特征逆深度传递给下层
			if (lvl < pyrLevelsUsed - 1)
				propagateDown(lvl + 1);

			resetPoints(lvl);
			Mat88f H, Hsc;
			Vec8f b, bsc;
			Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
			applyStep(lvl);

			float lambda = 0.1;
			float eps = 1e-4;
			int fails = 0;

			if (printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0] + resOld[1]) / resOld[2],
					(resOld[0] + resOld[1]) / resOld[2],
					0.0f);
				std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
			}

			int iteration = 0;
			while (true)
			{
				Mat88f Hl = H;
				float lambdaInv = 1 / (1 + lambda);
				for (int idx = 0; idx < 8; ++idx) Hl(idx, idx) *= (1 + lambda);
				Hl -= Hsc * lambdaInv;
				Vec8f bl = b - bsc * lambdaInv;

				Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
				bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

				Vec8f inc;
				if (setting_initFixAffine)
				{
					inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
					inc.tail<2>().setZero();
				}
				else
					inc = -(wM * (Hl.ldlt().solve(bl)));

				SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
				AffLight refToNew_aff_new = refToNew_aff_current;
				refToNew_aff_new.a += inc[6];
				refToNew_aff_new.b += inc[7];

				// 根据相机位姿、光度参数更新量计算特征逆深度更新量
				doStep(lvl, lambda, inc);

				Mat88f H_new, Hsc_new;
				Vec8f b_new, bsc_new;
				Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
				Vec3f regEnergy = calcEC(lvl);

				// 残差和为光度残差+位移量+逆深度残差
				float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
				float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

				bool accept = eTotalOld > eTotalNew;

				if (printDebug)
				{
					printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
					std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
				}

				if (accept)
				{
					// 位移比较大的时候，这此初始化比较可信
					if (resNew[1] == alphaK * numPoints[lvl])
						snapped = true;

					H = H_new;
					b = b_new;
					Hsc = Hsc_new;
					bsc = bsc_new;
					resOld = resNew;
					refToNew_aff_current = refToNew_aff_new;
					refToNew_current = refToNew_new;
					applyStep(lvl);
					optReg(lvl);
					lambda *= 0.5;			// 降低lambda加快收敛速度
					fails = 0;
					if (lambda < 0.0001) lambda = 0.0001;
				}
				else
				{
					fails++;
					lambda *= 4;			// 提高lambda使残差下降
					if (lambda > 10000) lambda = 10000;
				}

				bool quitOpt = false;

				// 迭代终止条件：1、增量比较小；2、迭代次数达到上限；3、一次迭代中连续失败两次以上
				if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
				{
					Mat88f H, Hsc;
					Vec8f b, bsc;
					quitOpt = true;
				}

				if (quitOpt) break;
				iteration++;
			}
			latestRes = resOld;
		}

		thisToNext = refToNew_current;
		thisToNext_aff = refToNew_aff_current;

		// 绘制在初始化中的跟踪结果
#ifdef SAVE_INITIALIZER_DATA
		int color = 0;
		int w = wG[0];
		int h = hG[0];
		int wh = w * h;

		MinimalImageB3* imgRef = new MinimalImageB3(wG[0], hG[0]);
		MinimalImageB3* imgNew = new MinimalImageB3(wG[0], hG[0]);

		Eigen::Vector3f* dIRef = firstFrame->dI;
		Eigen::Vector3f* dINew = newFrame->dI;

		Eigen::Vector2f ref2new_aff = Eigen::Vector2f(exp(thisToNext_aff.a), thisToNext_aff.b);

		// 绘制初始化handle中最新帧图像以及第一帧图像
		for (int it = 0; it < wh; ++it)
		{
			color = dIRef[it][0] * 0.8f;
			color = color * ref2new_aff[0] + ref2new_aff[1];
			if (color > 255) color = 255;
			imgRef->at(it) = Vec3b(color, color, color);

			color = dINew[it][0] * 0.8f;
			if (color > 255) color = 255;
			imgNew->at(it) = Vec3b(color, color, color);
		}

		// 绘制初始化handle中特征点信息搜索信息
		for (int lvl = 0; lvl < 1; ++lvl)
		{
			Pnt* pointl = points[lvl];
			int ptsNum = numPoints[lvl];
			int pixelIndex = 0, Ku = 0, Kv = 0;

			Mat33 KRKi = K[lvl] * thisToNext.rotationMatrix() * Ki[lvl];
			Vec3 Kt = K[lvl] * thisToNext.translation();

			for (int it = 0; it < ptsNum; ++it)
			{
				Ku = int(pointl[it].u) << lvl;
				Kv = int(pointl[it].v) << lvl;
				pixelIndex = Kv * w + Ku;
				imgRef->at(pixelIndex) = Vec3b(0, 255, 0);

				if (pointl[it].isGood)
				{
					Vec3 pt = KRKi * Vec3(pointl[it].u, pointl[it].v, 1) + Kt * pointl[it].idepth;
					Ku = int(pt[0] / pt[2]) << lvl;
					Kv = int(pt[1] / pt[2]) << lvl;

					if (Ku < 2 || Ku > w - 2 || Kv < 2 || Kv > h - 2) continue;
					pixelIndex = Kv * w + Ku;
					imgNew->at(pixelIndex) = Vec3b(0, 0, 255);
				}
			}
		}

		cv::Mat refImage = cv::Mat(imgRef->h, imgRef->w, CV_8UC3, imgRef->data);
		cv::Mat newImage = cv::Mat(imgNew->h, imgNew->w, CV_8UC3, imgNew->data);
#endif

		// 由上->下的金字塔优化结束后，逆深度值再由下到上平滑一遍
		for (int it = 0; it < pyrLevelsUsed - 1; ++it)
			propagateUp(it);

		frameID++;
		if (!snapped) snappedAt = 0;

		if (snapped && snappedAt == 0)
			snappedAt = frameID;

		// 始终将第零层金字塔图像对应的特征深度图传给显示线程显示
		debugPlotDepth(0, wraps);

		printf("snapped status: %s, snappedAt: %d frameID: %d\n",
			snapped == true ? "TRUE" : "FALSE", snappedAt, frameID);

		bool stepFrameFlag = !setting_useInitStep;
		if (setting_useInitStep)
			stepFrameFlag = frameID > snappedAt + setting_initStepFrames ? true : false;

		return snapped && stepFrameFlag;
	}

	/// <summary>
	/// 初始化handle中向显示线程传递特征深度图
	/// </summary>
	/// <param name="lvl">图像金字塔层级</param>
	/// <param name="wraps">显示线程handle</param>
	void CoarseInitializer::debugPlotDepth(int lvl, std::vector<IOWrap::Output3DWrapper*>& wraps)
	{
		bool needCall = false;
		for (IOWrap::Output3DWrapper* ow : wraps)
			needCall = needCall || ow->needPushDepthImage();
		if (!needCall) return;

		int wl = w[lvl], hl = h[lvl];
		Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

		MinimalImageB3 iRImg(wl, hl);

		for (int it = 0; it < wl * hl; ++it)
			iRImg.at(it) = Vec3b(colorRef[it][0], colorRef[it][0], colorRef[it][0]);

		int npts = numPoints[lvl];
		float nid = 0, sid = 0;

		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = points[lvl] + it;
			if (point->isGood)
			{
				nid++;
				sid += point->iR;
			}
		}
		float fac = nid / sid;

		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = points[lvl] + it;

			if (!point->isGood)
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));
			else
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
		}

		for (IOWrap::Output3DWrapper* ow : wraps)
			ow->pushDepthImage(&iRImg);
	}

	/// <summary>
	/// 计算单层金字塔中的光度残差、Hessian矩阵以及舒尔补
	/// </summary>
	/// <param name="lvl">待计算的金字塔层级</param>
	/// <param name="H_out">残差关于相机位姿和光度参数部分的Hessian</param>
	/// <param name="b_out">残差关于相机位姿和光度参数部分的b</param>
	/// <param name="H_out_sc">残差关于特征点逆深度部分的舒尔补项</param>
	/// <param name="b_out_sc">残差关于特征点逆深度部分的舒尔补项</param>
	/// <param name="refToNew">参考帧到关键帧的位姿变换</param>
	/// <param name="refToNew_aff">参考帧到关键帧的光度变换</param>
	/// <param name="plot">是否绘制中间结果</param>
	/// <returns></returns>
	Vec3f CoarseInitializer::calcResAndGS(int lvl, Mat88f& H_out, Vec8f& b_out,
		Mat88f& H_out_sc, Vec8f& b_out_sc, const SE3& refToNew, AffLight refToNew_aff, bool plot)
	{
		int wl = w[lvl], hl = h[lvl];
		Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
		Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

		Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
		Vec3f t = refToNew.translation().cast<float>();
		Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

		float fxl = fx[lvl];
		float fyl = fy[lvl];
		float cxl = cx[lvl];
		float cyl = cy[lvl];

		Accumulator11 E;
		acc9.initialize();
		E.initialize();

		int npts = numPoints[lvl];
		Pnt* ptsl = points[lvl];

		// 遍历提取的所有特征计算特征光度残差以及对应的雅可比
		for (int itPt = 0; itPt < npts; ++itPt)
		{
			Pnt* point = ptsl + itPt;
			point->maxstep = 1e10;

			// 光度残差过大或越界的特征点记录其上一次迭代的能量值
			if (!point->isGood)
			{
				E.updateSingle((float)(point->energy[0]));
				point->energy_new = point->energy;
				point->isGood_new = false;
				continue;
			}

			VecNRf dp0, dp1, dp2;		// 位置雅可比
			VecNRf dp3, dp4, dp5;		// 姿态雅可比
			VecNRf dp6, dp7;			// 光度参数雅可比
			VecNRf dd, r;				// 逆深度雅可比和残差
			JbBuffer_new[itPt].setZero();

			bool isGood = true;
			float energy = 0;

			// 对于单个特征为保证其鲁棒性，将特征周围的7个点与特征点一起计算残差和
			for (int idx = 0; idx < patternNum; ++idx)
			{
				int dx = patternP[idx][0];
				int dy = patternP[idx][1];

				Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
				float u = pt[0] / pt[2];						// 目标帧中归一化相机坐标系x坐标
				float v = pt[1] / pt[2];						// 目标帧中归一化相机坐标系y坐标
				float Ku = fxl * u + cxl;						// 目标帧中像素坐标系下的u轴坐标
				float Kv = fyl * v + cyl;						// 目标帧中像素坐标系下的v轴坐标
				float new_idepth = point->idepth_new / pt[2];	// 特征点在目标帧相机坐标系下的逆深度

				if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
				{
					isGood = false;
					break;
				}

				Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
				float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

				if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
				{
					isGood = false;
					break;
				}

				float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
				float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
				energy += hw * residual * residual * (2 - hw);

				float dxdd = (t[0] - t[2] * u) / pt[2];
				float dydd = (t[1] - t[2] * v) / pt[2];

				if (hw < 1) hw = sqrtf(hw);
				float dxInterp = hw * hitColor[1] * fxl;
				float dyInterp = hw * hitColor[2] * fyl;

				// 计算光度残差相对于位姿的雅可比：前三项为平移后三项为旋转[检查了，没问题]
				dp0[idx] = new_idepth * dxInterp;
				dp1[idx] = new_idepth * dyInterp;
				dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
				dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
				dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
				dp5[idx] = -v * dxInterp + u * dyInterp;

				// 计算光度残差相对于光度参数的雅可比：第一项为a21第二项为b21
				dp6[idx] = -hw * r2new_aff[0] * rlR;
				dp7[idx] = -hw * 1;

				// 计算光度残差相对于参考帧特征逆深度的雅可比[检查了，没问题]
				dd[idx] = dxInterp * dxdd + dyInterp * dydd;

				// 加权残差项
				r[idx] = hw * residual;

				// [dxdd*fxl, dydd*fyl]为像素坐标对逆深度的雅可比，表征了像素变化随逆深度变化的情况
				// 此处设置像素最大变化量为1个像素，计算像素变化一个像素时逆深度的变化量为maxstep
				float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
				if (maxstep < point->maxstep) point->maxstep = maxstep;

				// immediately compute dp*dd' and dd*dd' in JbBuffer1.
				JbBuffer_new[itPt][0] += dp0[idx] * dd[idx];
				JbBuffer_new[itPt][1] += dp1[idx] * dd[idx];
				JbBuffer_new[itPt][2] += dp2[idx] * dd[idx];
				JbBuffer_new[itPt][3] += dp3[idx] * dd[idx];
				JbBuffer_new[itPt][4] += dp4[idx] * dd[idx];
				JbBuffer_new[itPt][5] += dp5[idx] * dd[idx];
				JbBuffer_new[itPt][6] += dp6[idx] * dd[idx];
				JbBuffer_new[itPt][7] += dp7[idx] * dd[idx];
				JbBuffer_new[itPt][8] += r[idx] * dd[idx];
				JbBuffer_new[itPt][9] += dd[idx] * dd[idx];
			}

			// 光度误差太大时该特征不参加优化
			if (!isGood || energy > point->outlierTH * 20)
			{
				E.updateSingle((float)(point->energy[0]));
				point->isGood_new = false;
				point->energy_new = point->energy;
				continue;
			}

			E.updateSingle(energy);
			point->isGood_new = true;
			point->energy_new[0] = energy;

			// 计算光度残差相对于相机位姿以及光度参数的雅可比
			// 计算得到的结果为9x9的对称矩阵，前8x8维度矩阵为Hessian，后8x1为b
			for (int i = 0; i + 3 < patternNum; i += 4)
				acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0)) + i),
					_mm_load_ps(((float*)(&dp1)) + i),
					_mm_load_ps(((float*)(&dp2)) + i),
					_mm_load_ps(((float*)(&dp3)) + i),
					_mm_load_ps(((float*)(&dp4)) + i),
					_mm_load_ps(((float*)(&dp5)) + i),
					_mm_load_ps(((float*)(&dp6)) + i),
					_mm_load_ps(((float*)(&dp7)) + i),
					_mm_load_ps(((float*)(&r)) + i));

			// 对于patternNum不为4的倍数的情况，多出来的不能用SSE只能逐个累加
			for (int i = ((patternNum >> 2) << 2); i < patternNum; ++i)
				acc9.updateSingle(
					(float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
					(float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
					(float)r[i]);
		}

		E.finish();		// 完成计算优化问题的Huber约束下的能量值
		acc9.finish();	// 完成计算相对于位姿以及光度参数雅可比

		// 添加L2正则项避免优化问题出现过拟合或欠拟合的情况
		Accumulator11 EAlpha;
		EAlpha.initialize();
		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = ptsl + it;
			if (!point->isGood_new)
			{
				EAlpha.updateSingle((float)(point->energy[1]));
			}
			else
			{
				point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
				EAlpha.updateSingle((float)(point->energy_new[1]));
			}
		}
		EAlpha.finish();
		float alphaOpt = 0;
		float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);		// 平移越大初始化效果越好

		// 位移过大时添加的L2正则项
		if (alphaEnergy > alphaK * npts)
		{
			alphaOpt = 0;
			alphaEnergy = alphaK * npts;
		}
		// 位移较小时添加的L2正则项
		else
		{
			alphaOpt = alphaW;
		}

		// 计算关于相机位姿和光度参数的舒尔补项
		acc9SC.initialize();
		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = ptsl + it;
			if (!point->isGood_new)
				continue;

			point->lastHessian_new = JbBuffer_new[it][9];

			JbBuffer_new[it][8] += alphaOpt * (point->idepth_new - 1);
			JbBuffer_new[it][9] += alphaOpt;

			// 当位移比较大的时候添加的正则项在Jacobian中也需要加上对应偏导
			if (alphaOpt == 0)
			{
				JbBuffer_new[it][8] += couplingWeight * (point->idepth_new - point->iR);
				JbBuffer_new[it][9] += couplingWeight;
			}

			// 计算大Hessian矩阵边缘化特征逆深度后的舒尔补项
			JbBuffer_new[it][9] = 1 / (1 + JbBuffer_new[it][9]);
			acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[it][0], (float)JbBuffer_new[it][1], (float)JbBuffer_new[it][2], (float)JbBuffer_new[it][3],
				(float)JbBuffer_new[it][4], (float)JbBuffer_new[it][5], (float)JbBuffer_new[it][6], (float)JbBuffer_new[it][7],
				(float)JbBuffer_new[it][8], (float)JbBuffer_new[it][9]);
		}
		acc9SC.finish();

		H_out = acc9.H.topLeftCorner<8, 8>();// / acc9.num;
		b_out = acc9.H.topRightCorner<8, 1>();// / acc9.num;
		H_out_sc = acc9SC.H.topLeftCorner<8, 8>();// / acc9.num;
		b_out_sc = acc9SC.H.topRightCorner<8, 1>();// / acc9.num;

		H_out(0, 0) += alphaOpt * npts;
		H_out(1, 1) += alphaOpt * npts;
		H_out(2, 2) += alphaOpt * npts;

		//Vec3f tlog = refToNew.translation().cast<float>();
		Vec3f tlog = refToNew.log().head<3>().cast<float>();
		b_out[0] += tlog[0] * alphaOpt * npts;
		b_out[1] += tlog[1] * alphaOpt * npts;
		b_out[2] += tlog[2] * alphaOpt * npts;

		return Vec3f(E.A, alphaEnergy, E.num);
	}

	float CoarseInitializer::rescale()
	{
		float factor = 20 * thisToNext.translation().norm();
		//	float factori = 1.0f/factor;
		//	float factori2 = factori*factori;
		//
		//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		//	{
		//		int npts = numPoints[lvl];
		//		Pnt* ptsl = points[lvl];
		//		for(int i=0;i<npts;i++)
		//		{
		//			ptsl[i].iR *= factor;
		//			ptsl[i].idepth_new *= factor;
		//			ptsl[i].lastHessian *= factori2;
		//		}
		//	}
		//	thisToNext.translation() *= factori;
		return factor;
	}

	/// <summary>
	/// 计算一次优化迭代前后逆深度残差值
	/// </summary>
	/// <param name="lvl">待计算的金字塔层级</param>
	/// <returns>逆深度残差和：[0]迭代前逆深度残差、
	///						   [1]迭代后逆深度残差、
	///						   [2]用于计算残差的特征数量
	/// </returns>
	Vec3f CoarseInitializer::calcEC(int lvl)
	{
		if (!snapped)
			return Vec3f(0, 0, numPoints[lvl]);

		AccumulatorX<2> E;
		E.initialize();
		int npts = numPoints[lvl];

		// 统计逆深度优化前后的残差值
		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = points[lvl] + it;
			if (!point->isGood_new) continue;
			float rOld = (point->idepth - point->iR);
			float rNew = (point->idepth_new - point->iR);
			E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));
		}
		E.finish();

		// [0]迭代前逆深度残差、[1]迭代后逆深度残差、[2]用于计算残差的特征数量
		return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
	}

	/// <summary>
	/// 同层金字塔特征逆深度优化，按照邻近特征逆深度中值与该特征逆深度融合策略优化
	/// </summary>
	/// <param name="lvl">待优化特征逆深度的金字塔层级</param>
	void CoarseInitializer::optReg(int lvl)
	{
		int npts = numPoints[lvl];
		Pnt* ptsl = points[lvl];

		if (!snapped)
		{
			for (int it = 0; it < npts; ++it)
				ptsl[it].iR = 1;
			return;
		}

		for (int it = 0; it < npts; ++it)
		{
			Pnt* point = ptsl + it;
			if (!point->isGood) continue;

			float idnn[10];
			int nnn = 0;
			for (int j = 0; j < 10; j++)
			{
				if (point->neighbours[j] == -1) continue;
				Pnt* other = ptsl + point->neighbours[j];
				if (!other->isGood) continue;
				idnn[nnn] = other->iR;
				nnn++;
			}

			// 取邻近点逆深度中值与当前特征逆深度融合作为该特征的逆深度均值
			if (nnn > 2)
			{
				std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
				point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
			}
		}
	}

	void CoarseInitializer::propagateUp(int srcLvl)
	{
		assert(srcLvl + 1 < pyrLevelsUsed);

		int nptss = numPoints[srcLvl];
		int nptst = numPoints[srcLvl + 1];
		Pnt* ptss = points[srcLvl];
		Pnt* ptst = points[srcLvl + 1];

		for (int it = 0; it < nptst; ++it)
		{
			Pnt* parent = ptst + it;
			parent->iR = 0;
			parent->iRSumNum = 0;
		}

		for (int it = 0; it < nptss; ++it)
		{
			Pnt* point = ptss + it;
			if (!point->isGood) continue;

			Pnt* parent = ptst + point->parent;
			parent->iR += point->iR * point->lastHessian;
			parent->iRSumNum += point->lastHessian;
		}

		for (int it = 0; it < nptst; ++it)
		{
			Pnt* parent = ptst + it;
			if (parent->iRSumNum > 0)
			{
				parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
				parent->isGood = true;
			}
		}

		optReg(srcLvl + 1);
	}

	void CoarseInitializer::propagateDown(int srcLvl)
	{
		assert(srcLvl > 0);
		// set idepth of target

		int nptst = numPoints[srcLvl - 1];
		Pnt* ptss = points[srcLvl];
		Pnt* ptst = points[srcLvl - 1];

		for (int i = 0; i < nptst; i++)
		{
			Pnt* point = ptst + i;
			Pnt* parent = ptss + point->parent;

			if (!parent->isGood || parent->lastHessian < 0.1) continue;
			if (!point->isGood)
			{
				point->iR = point->idepth = point->idepth_new = parent->iR;
				point->isGood = true;
				point->lastHessian = 0;
			}
			else
			{
				float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
				point->iR = point->idepth = point->idepth_new = newiR;
			}
		}
		optReg(srcLvl - 1);
	}

	/// <summary>
	/// 计算传入图像数据的梯度值，但是DSO中并没有使用这个函数
	/// </summary>
	/// <param name="data">输入不同金字塔层图像数据</param>
	void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
	{
		for (int lvl = 1; lvl < pyrLevelsUsed; ++lvl)
		{
			int lvlm1 = lvl - 1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			Eigen::Vector3f* dINew_l = data[lvl];
			Eigen::Vector3f* dINew_lm = data[lvlm1];

			for (int y = 0; y < hl; ++y)
				for (int x = 0; x < wl; ++x)
					dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
						dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
						dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
						dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

			for (int idx = wl; idx < wl * (hl - 1); ++idx)
			{
				dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
				dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
			}
		}
	}

	/// <summary>
	/// 初始化handle中设置第一帧图像并在图像不同金字塔层中提取特征点
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	/// <param name="newFrameHessian">进入系统的最新帧</param>
	void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrame)
	{
		makeK(HCalib);
		firstFrame = newFrame;

		PixelSelector sel(w[0], h[0]);

		float* statusMap = new float[w[0] * h[0]];
		bool* statusMapB = new bool[w[0] * h[0]];
		//float densities[] = { 0.03,0.05,0.15,0.5,1 };
		float densities[] = { 1,1,1,1,1 };

		// 在各层金字塔中选点并初始化各个点的深度值
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			int npts = 0;
			sel.currentPotential = 3;

			// 第零层金字塔按照PixelSelector类去选特征其他层金字塔按照另外的函数makePixelStatus选择特征
			if (lvl == 0)
				npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, true, 2);
			else
			{
				npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);
				debugPlotFeatureDetect(firstFrame->dIp[lvl], lvl, statusMapB, w[lvl], h[lvl]);
			}

			SAFE_DELETE(points[lvl]);
			points[lvl] = new Pnt[npts];

			// 初始化所有点的逆深度值为1
			Pnt* pl = points[lvl];
			int wl = w[lvl], hl = h[lvl];
			int pixelIndex = 0, nl = 0;

			for (int y = patternPadding + 1; y < hl - patternPadding - 2; ++y)
			{
				for (int x = patternPadding + 1; x < wl - patternPadding - 2; ++x)
				{
					pixelIndex = x + y * wl;
					if ((lvl != 0 && statusMapB[pixelIndex]) ||
						(lvl == 0 && statusMap[pixelIndex] != 0))
					{
						pl[nl].u = x + 0.1;
						pl[nl].v = y + 0.1;
						pl[nl].idepth = 1;
						pl[nl].iR = 1;
						pl[nl].isGood = true;
						pl[nl].energy.setZero();
						pl[nl].lastHessian = 0;
						pl[nl].lastHessian_new = 0;
						pl[nl].my_type = (lvl != 0) ? 1 : statusMap[pixelIndex];

						int dx = 0, dy = 0;
						float sumGrad2 = 0, absgrad = 0;
						Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + pixelIndex;

						// 计算所选特征点与周围点的梯度值和用于计算特征光度阈值
						for (int idx = 0; idx < patternNum; idx++)
						{
							dx = patternP[idx][0];
							dy = patternP[idx][1];
							absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
							sumGrad2 += absgrad;
						}

						pl[nl++].outlierTH = patternNum * setting_outlierTH;
						assert(nl <= npts);
					}
				}
			}

			numPoints[lvl] = nl;
		}

		SAFE_DELETE(statusMap, true);
		SAFE_DELETE(statusMapB, true);

		// 构造金字塔中所提取特征的空间拓扑结构
		makeNN();

		thisToNext = SE3();
		snapped = false;
		frameID = snappedAt = 0;

		for (int it = 0; it < pyrLevelsUsed; ++it)
			dGrads[it].setZero();
	}

	/// <summary>
	/// 初始化handle中设置第一帧图像并且在该图像中选取特征点
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	/// <param name="newFrameHessian">进入系统的最新帧</param>
	/// <param name="newFrameHessianRight">进入系统的最新帧对应的右目图像</param>
	void CoarseInitializer::setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessianRight)
	{
		makeK(HCalib);
		firstFrame = newFrameHessian;
		firstFrameRight = newFrameHessianRight;

		PixelSelector sel(w[0], h[0]);

		float* statusMap = new float[w[0] * h[0]];
		bool* statusMapB = new bool[w[0] * h[0]];
		float densities[] = { 0.03,0.05,0.15,0.5,1 };

		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
			idepth[lvl] = new float[w[lvl] * h[lvl]]{ 0 };

		// 在各层金字塔中选点并初始化各个点的深度值
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			sel.currentPotential = 3;
			int npts = 0;
			if (lvl == 0)
				npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
			else
				npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

			if (points[lvl] != 0) delete[] points[lvl];
			points[lvl] = new Pnt[npts];

			Pnt* pl = points[lvl];
			int wl = w[lvl], hl = h[lvl], nl = 0;

			// 初始化所有点的逆深度值为1
			for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
			{
				for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++)
				{
					if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0))
					{
						pl[nl].u = x + 0.1;
						pl[nl].v = y + 0.1;
						pl[nl].idepth = 1;
						pl[nl].iR = 1;
						pl[nl].isGood = true;
						pl[nl].energy.setZero();
						pl[nl].lastHessian = 0;
						pl[nl].lastHessian_new = 0;
						pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

						int dx = 0, dy = 0;
						float absgrad = 0, sumGrad2 = 0;
						Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];

						for (int idx = 0; idx < patternNum; idx++)
						{
							dx = patternP[idx][0];
							dy = patternP[idx][1];
							absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
							sumGrad2 += absgrad;
						}

						pl[nl++].outlierTH = patternNum * setting_outlierTH;
						assert(nl <= npts);
					}
				}
			}

			numPoints[lvl] = nl;
		}
		SAFE_DELETE(statusMap, true);
		SAFE_DELETE(statusMapB, true);

		makeNN();

		thisToNext = SE3();
		snapped = false;
		frameID = snappedAt = 0;

		for (int it = 0; it < pyrLevelsUsed; ++it)
			dGrads[it].setZero();
	}

	void CoarseInitializer::resetPoints(int lvl)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		for (int it = 0; it < npts; ++it)
		{
			pts[it].energy.setZero();
			pts[it].idepth_new = pts[it].idepth;

			// 若需要重置底层金字塔，则需要使用领域特征逆深度进行计算
			if (lvl == pyrLevelsUsed - 1 && !pts[it].isGood)
			{
				float snd = 0, sn = 0;
				for (int n = 0; n < 10; n++)
				{
					if (pts[it].neighbours[n] == -1 || !pts[pts[it].neighbours[n]].isGood) continue;
					snd += pts[pts[it].neighbours[n]].iR;
					sn += 1;
				}

				if (sn > 0)
				{
					pts[it].isGood = true;
					pts[it].iR = pts[it].idepth = pts[it].idepth_new = snd / sn;
				}
			}
		}
	}

	/// <summary>
	/// 对每层金字塔中选中的特征计算其逆深度更新值
	/// </summary>
	/// <param name="lvl">金字塔层级</param>
	/// <param name="lambda">优化收敛控制因子</param>
	/// <param name="inc">关键帧位姿和光度参数增益</param>
	void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
	{
		const float maxPixelStep = 0.25;
		const float idMaxStep = 1e10;
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		for (int it = 0; it < npts; ++it)
		{
			if (!pts[it].isGood) continue;

			float b = JbBuffer[it][8] + JbBuffer[it].head<8>().dot(inc);
			float step = -b * JbBuffer[it][9] / (1 + lambda);

			// 控制逆深度更新量的步进范围避免ZIGZAG现象
			float maxstep = maxPixelStep * pts[it].maxstep;
			if (maxstep > idMaxStep) maxstep = idMaxStep;

			if (step > maxstep) step = maxstep;
			if (step < -maxstep) step = -maxstep;

			// 控制逆深度的范围，太近或太远逆深度并不可信
			float newIdepth = pts[it].idepth + step;
			if (newIdepth < 1e-3) newIdepth = 1e-3;
			if (newIdepth > 50) newIdepth = 50;
			pts[it].idepth_new = newIdepth;
		}
	}

	/// <summary>
	/// 接受当前优化迭代步骤并调整相关成员变量
	/// </summary>
	/// <param name="lvl">金字塔层级</param>
	void CoarseInitializer::applyStep(int lvl)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		for (int it = 0; it < npts; ++it)
		{
			if (!pts[it].isGood)
			{
				pts[it].idepth = pts[it].idepth_new = pts[it].iR;
				continue;
			}
			pts[it].energy = pts[it].energy_new;
			pts[it].isGood = pts[it].isGood_new;
			pts[it].idepth = pts[it].idepth_new;
			pts[it].lastHessian = pts[it].lastHessian_new;
		}

		Vec10f* JbTmp = JbBuffer;
		JbBuffer = JbBuffer_new;
		JbBuffer_new = JbTmp;
	}

	/// <summary>
	/// 设置初始化中不同金字塔层相机内参相关信息
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	void CoarseInitializer::makeK(CalibHessian* HCalib)
	{
		w[0] = wG[0];
		h[0] = hG[0];

		fx[0] = HCalib->fxl();
		fy[0] = HCalib->fyl();
		cx[0] = HCalib->cxl();
		cy[0] = HCalib->cyl();

		for (int level = 1; level < pyrLevelsUsed; ++level)
		{
			w[level] = w[0] >> level;
			h[level] = h[0] >> level;
			fx[level] = fx[level - 1] * 0.5;
			fy[level] = fy[level - 1] * 0.5;
			cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
		}

		for (int level = 0; level < pyrLevelsUsed; ++level)
		{
			K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
			Ki[level] = K[level].inverse();
			fxi[level] = Ki[level](0, 0);
			fyi[level] = Ki[level](1, 1);
			cxi[level] = Ki[level](0, 2);
			cyi[level] = Ki[level](1, 2);
		}
	}

	/// <summary>
	/// 构造不同金字塔选取的特征点之间的空间拓扑结构  
	/// 所谓拓扑结构指同层金字塔中搜索邻近点并在上层金字塔中搜索最邻近点作为父点
	/// </summary>
	void CoarseInitializer::makeNN()
	{
		const int nn = 10;
		const float NNDistFactor = 0.05;

		typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
			FLANNPointcloud, 2> KDTree;

		// 使用金字塔中得到的特征数据构造八叉树
		FLANNPointcloud pcs[PYR_LEVELS];
		KDTree* indexes[PYR_LEVELS];
		for (int it = 0; it < pyrLevelsUsed; ++it)
		{
			pcs[it] = FLANNPointcloud(numPoints[it], points[it]);
			indexes[it] = new KDTree(2, pcs[it], nanoflann::KDTreeSingleIndexAdaptorParams(5));
			indexes[it]->buildIndex();
		}

		// 使用八叉树搜索最邻近点
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			Pnt* pts = points[lvl];
			int npts = numPoints[lvl];

			int ret_index[nn];
			float ret_dist[nn];
			nanoflann::KNNResultSet<float, int, int> resultSet(nn);
			nanoflann::KNNResultSet<float, int, int> resultSet1(1);

			for (int it = 0; it < npts; ++it)
			{
				resultSet.init(ret_index, ret_dist);
				Vec2f pt = Vec2f(pts[it].u, pts[it].v);
				indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());

				// 同层金字塔中搜索10个最邻近点并计算反距离权重值
				int idx = 0;
				float sumDF = 0, df = 0;
				for (int k = 0; k < nn; ++k)
				{
					pts[it].neighbours[idx] = ret_index[k];
					df = expf(-ret_dist[k] * NNDistFactor);
					pts[it].neighboursDist[idx] = df;
					assert(ret_index[k] >= 0 && ret_index[k] < npts);
					sumDF += df; idx++;
				}

				float sumDFFactor = 10.0 / sumDF;
				for (int k = 0; k < nn; k++)
					pts[it].neighboursDist[k] *= sumDFFactor;

				// 在上一层金字塔中搜索最邻近点作为当前遍历特征的父特征
				if (lvl < pyrLevelsUsed - 1)
				{
					resultSet1.init(ret_index, ret_dist);
					pt = pt * 0.5f - Vec2f(0.25f, 0.25f); // TODO：0.25精度问题
					indexes[lvl + 1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

					pts[it].parent = ret_index[0];
					pts[it].parentDist = expf(-ret_dist[0] * NNDistFactor);

					assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
				}
				else
				{
					pts[it].parent = -1;
					pts[it].parentDist = -1;
				}
			}
		}

		for (int it = 0; it < pyrLevelsUsed; ++it)
		{
			delete indexes[it];
			indexes[it] = nullptr;
		}
	}
}