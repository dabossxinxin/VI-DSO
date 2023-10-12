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
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			points[lvl] = 0;
			numPoints[lvl] = 0;
		}

		JbBuffer = new Vec10f[ww*hh];
		JbBuffer_new = new Vec10f[ww*hh];

		frameID = -1;
		fixAffine = true;
		printDebug = false;

		wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
		wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
		wM.diagonal()[6] = SCALE_A;
		wM.diagonal()[7] = SCALE_B;
	}

	CoarseInitializer::~CoarseInitializer()
	{
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			if (points[lvl] != 0) delete[] points[lvl];
		}

		delete[] JbBuffer;
		delete[] JbBuffer_new;
	}

	/// <summary>
	/// 初始化中跟踪最新帧
	/// </summary>
	/// <param name="newFrame">最新帧数据</param>
	/// <param name="wraps">Pangolin显示器</param>
	/// <returns></returns>
	bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
	{
		newFrame = newFrameHessian;

		for (IOWrap::Output3DWrapper* ow : wraps)
			ow->pushLiveFrame(newFrameHessian);

		int maxIterations[] = { 5,5,10,30,50 };

		// 设置正则项相关参数
		alphaK = 2.5*2.5;	//*freeDebugParam1*freeDebugParam1;
		alphaW = 150 * 150;	//*freeDebugParam2*freeDebugParam2;
		regWeight = 0.8;	//*freeDebugParam4;
		couplingWeight = 1;	//*freeDebugParam5;

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

		SE3 refToNew_current = thisToNext;
		AffLight refToNew_aff_current = thisToNext_aff;

		if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
			refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0); // coarse approximation.

		Vec3f latestRes = Vec3f::Zero();

		// 从金字塔顶层向金字塔底层逐层优化各项参数
		for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
		{
			// 将上层优化得到的特征逆深度传递给下层
			if (lvl < pyrLevelsUsed - 1)
				propagateDown(lvl + 1);

			resetPoints(lvl);
			Mat88f H, Hsc; Vec8f b, bsc;
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
				for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
				Hl -= Hsc * (1 / (1 + lambda));
				Vec8f bl = b - bsc * (1 / (1 + lambda));

				Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
				bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

				Vec8f inc;
				if (fixAffine)
				{
					inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
					inc.tail<2>().setZero();             
				}
				else
					inc = -(wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

				SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
				AffLight refToNew_aff_new = refToNew_aff_current;
				refToNew_aff_new.a += inc[6];
				refToNew_aff_new.b += inc[7];

				// 根据相机位姿、光度参数更新量计算特征逆深度更新量
				doStep(lvl, lambda, inc);

				Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
				Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
				Vec3f regEnergy = calcEC(lvl);

				// 残差和为光度残差+L2正则化项残差+逆深度残差
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

		// 由上->下的金字塔优化结束后，逆深度值再由下到上平滑一遍
		for (int it = 0; it < pyrLevelsUsed - 1; ++it)
			propagateUp(it);

		frameID++;
		if (!snapped) snappedAt = 0;

		if (snapped && snappedAt == 0)
			snappedAt = frameID;

		debugPlot(0, wraps);

		printf("snapped status: %s, snappedAt: %d frameID: %d\n",
			snapped == true ? "TRUE" : "FALSE", snappedAt, frameID);

		// 检测到位移较大的两帧并且之后五帧都可以跟踪上
		return snapped && frameID > snappedAt + 5;
	}

	void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
	{
		bool needCall = false;
		for (IOWrap::Output3DWrapper* ow : wraps)
			needCall = needCall || ow->needPushDepthImage();
		if (!needCall) return;

		int wl = w[lvl], hl = h[lvl];
		Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

		MinimalImageB3 iRImg(wl, hl);

		for (int i = 0; i < wl*hl; i++)
			iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);

		int npts = numPoints[lvl];

		float nid = 0, sid = 0;
		for (int i = 0; i < npts; i++)
		{
			Pnt* point = points[lvl] + i;
			if (point->isGood)
			{
				nid++;
				sid += point->iR;
			}
		}
		float fac = nid / sid;

		for (int i = 0; i < npts; i++)
		{
			Pnt* point = points[lvl] + i;

			if (!point->isGood)
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

			else
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR*fac));
		}

		//IOWrap::displayImage("idepth-R", &iRImg, false);
		for (IOWrap::Output3DWrapper* ow : wraps)
			ow->pushDepthImage(&iRImg);
	}

	/// <summary>
	/// 计算单层金字塔中的光度残差、Hessian矩阵以及舒尔补
	/// </summary>
	/// <param name="lvl">待计算的金字塔层级</param>
	/// <param name="H_out"></param>
	/// <param name="b_out"></param>
	/// <param name="H_out_sc"></param>
	/// <param name="b_out_sc"></param>
	/// <param name="refToNew"></param>
	/// <param name="refToNew_aff"></param>
	/// <param name="plot"></param>
	/// <returns></returns>
	Vec3f CoarseInitializer::calcResAndGS(int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew, AffLight refToNew_aff, bool plot)
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
		for (int i = 0; i < npts; i++)
		{
			Pnt* point = ptsl + i;
			point->maxstep = 1e10;

			if (!point->isGood)
			{
				E.updateSingle((float)(point->energy[0]));
				point->energy_new = point->energy;
				point->isGood_new = false;
				continue;
			}

			VecNRf dp0;
			VecNRf dp1;
			VecNRf dp2;
			VecNRf dp3;
			VecNRf dp4;
			VecNRf dp5;
			VecNRf dp6;
			VecNRf dp7;
			VecNRf dd;
			VecNRf r;
			JbBuffer_new[i].setZero();

			bool isGood = true;
			float energy = 0;

			// 对于单个特征为保证其鲁棒性，将特征周围的7个点与特征点一起计算残差和
			for (int idx = 0; idx < patternNum; idx++)
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

				// 计算光度残差相对于位姿的雅可比：前三项为平移后三项为旋转
				dp0[idx] = new_idepth * dxInterp;
				dp1[idx] = new_idepth * dyInterp;
				dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
				dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
				dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
				dp5[idx] = -v * dxInterp + u * dyInterp;

				// 计算光度残差相对于光度参数的雅可比：第一项为a21第二项为b21
				dp6[idx] = -hw * r2new_aff[0] * rlR;
				dp7[idx] = -hw * 1;

				// 计算光度残差相对于参考帧特征逆深度的雅可比
				dd[idx] = dxInterp * dxdd + dyInterp * dydd;

				// 加权残差项
				r[idx] = hw * residual;

				// [dxdd*fxl, dydd*fyl]为像素坐标对逆深度的雅可比，表征了像素变化随逆深度变化的情况
				// 此处设置像素最大变化量为1个像素，计算像素变化一个像素时逆深度的变化量为maxstep
				float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
				if (maxstep < point->maxstep) point->maxstep = maxstep;

				// immediately compute dp*dd' and dd*dd' in JbBuffer1.
				JbBuffer_new[i][0] += dp0[idx] * dd[idx];
				JbBuffer_new[i][1] += dp1[idx] * dd[idx];
				JbBuffer_new[i][2] += dp2[idx] * dd[idx];
				JbBuffer_new[i][3] += dp3[idx] * dd[idx];
				JbBuffer_new[i][4] += dp4[idx] * dd[idx];
				JbBuffer_new[i][5] += dp5[idx] * dd[idx];
				JbBuffer_new[i][6] += dp6[idx] * dd[idx];
				JbBuffer_new[i][7] += dp7[idx] * dd[idx];
				JbBuffer_new[i][8] += r[idx] * dd[idx];
				JbBuffer_new[i][9] += dd[idx] * dd[idx];
			}

			// 光度误差太大时该特征不参加优化
			if (!isGood || energy > point->outlierTH * 20)
			{
				E.updateSingle((float)(point->energy[0]));
				point->isGood_new = false;
				point->energy_new = point->energy;
				continue;
			}

			// add into energy.
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
				E.updateSingle((float)(point->energy[1]));
			}
			else
			{
				point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
				E.updateSingle((float)(point->energy_new[1]));
			}
		}
		EAlpha.finish();
		float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

		float alphaOpt;

		// 位移过大时添加的L2正则项
		if (alphaEnergy > alphaK*npts)
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

			JbBuffer_new[it][9] = 1 / (1 + JbBuffer_new[it][9]);	// 相对于逆深度的Hessian取逆
			acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[it][0], (float)JbBuffer_new[it][1], (float)JbBuffer_new[it][2], (float)JbBuffer_new[it][3],
				(float)JbBuffer_new[it][4], (float)JbBuffer_new[it][5], (float)JbBuffer_new[it][6], (float)JbBuffer_new[it][7],
				(float)JbBuffer_new[it][8], (float)JbBuffer_new[it][9]);
		}
		acc9SC.finish();

		//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
		H_out = acc9.H.topLeftCorner<8, 8>();// / acc9.num;
		b_out = acc9.H.topRightCorner<8, 1>();// / acc9.num;
		H_out_sc = acc9SC.H.topLeftCorner<8, 8>();// / acc9.num;
		b_out_sc = acc9SC.H.topRightCorner<8, 1>();// / acc9.num;

		H_out(0, 0) += alphaOpt * npts;
		H_out(1, 1) += alphaOpt * npts;
		H_out(2, 2) += alphaOpt * npts;

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
				float newiR = (point->iR*point->lastHessian * 2 + parent->iR*parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
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
					dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y*wlm1][0] +
						dINew_lm[2 * x + 1 + 2 * y*wlm1][0] +
						dINew_lm[2 * x + 2 * y*wlm1 + wlm1][0] +
						dINew_lm[2 * x + 1 + 2 * y*wlm1 + wlm1][0]);

			for (int idx = wl; idx < wl*(hl - 1); ++idx)
			{
				dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
				dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
			}
		}
	}

	/// <summary>
	/// DSO初始化中设置第一帧图像并在图像不同金字塔层中选择特征点
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	/// <param name="newFrameHessian">进入DSO初始化器中的视觉帧</param>
	void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrame)
	{
		makeK(HCalib);
		firstFrame = newFrame;

		PixelSelector sel(w[0], h[0]);

		float* statusMap = new float[w[0] * h[0]];
		bool* statusMapB = new bool[w[0] * h[0]];
		float densities[] = { 0.03,0.05,0.15,0.5,1 };

		// 在各层金字塔中选点并初始化各个点的深度值
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			int npts = 0;
			sel.currentPotential = 3;
			
			// PixelSelector选点
			if (lvl == 0) npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
			else npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

			if (points[lvl] != 0) delete[] points[lvl];
			points[lvl] = new Pnt[npts];

			// 初始化所有点的逆深度值为1
			Pnt* pl = points[lvl];
			int wl = w[lvl], hl = h[lvl], nl = 0;
		
			for (int y = patternPadding + 1; y < hl - patternPadding - 2; ++y)
			{
				for (int x = patternPadding + 1; x < wl - patternPadding - 2; ++x)
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
						float sumGrad2 = 0, absgrad = 0;
						Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
						
						// 计算所选特征点与周围点的梯度值和
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

		delete[] statusMap; statusMap = NULL;
		delete[] statusMapB; statusMapB = NULL;

		// 构造金字塔中所提取特征的空间拓扑结构
		makeNN();

		thisToNext = SE3();
		snapped = false;
		frameID = snappedAt = 0;

		for (int it = 0; it < pyrLevelsUsed; ++it)
			dGrads[it].setZero();
	}

	void CoarseInitializer::setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_right)
	{
		makeK(HCalib);
		firstFrame = newFrameHessian;
		firstFrame_right = newFrameHessian_right;

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

						//float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
						//pl[nl].outlierTH = patternNum*gth*gth;

						pl[nl++].outlierTH = patternNum * setting_outlierTH;
						assert(nl <= npts);
					}
				}
			}

			// 		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
			// 		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
			// 		{
			// 			if(lvl==0 && statusMap[x+y*wl] != 0) {
			// 				ImmaturePoint* pt = new ImmaturePoint(x, y, firstFrame, statusMap[x+y*wl], HCalib);
			// 				pt->u_stereo = pt->u;
			// 				pt->v_stereo = pt->v;
			// 				pt->idepth_min_stereo = 0;
			// 				pt->idepth_max_stereo = NAN;
			// 				ImmaturePointStatus stat = pt->traceStereo(firstFrame_right, HCalib, true);
			// 				if(stat==ImmaturePointStatus::IPS_GOOD) {
			// 					pl[nl].u = x;
			// 					pl[nl].v = y;
			// 
			// 					pl[nl].idepth = pt->idepth_stereo;
			// 					pl[nl].iR = pt->idepth_stereo;
			// 
			// 					pl[nl].isGood=true;
			// 					pl[nl].energy.setZero();
			// 					pl[nl].lastHessian=0;
			// 					pl[nl].lastHessian_new=0;
			// 					pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
			// 					idepth[0][x+wl*y] = pt->idepth_stereo;
			// 
			// 					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
			// 					float sumGrad2=0;
			// 					for(int idx=0;idx<patternNum;idx++)
			// 					{
			// 						int dx = patternP[idx][0];
			// 						int dy = patternP[idx][1];
			// 						float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
			// 						sumGrad2 += absgrad;
			// 					}
			// 
			// 					pl[nl].outlierTH = patternNum*setting_outlierTH;
			// 					nl++;
			// 					assert(nl <= npts);
			// 				} 
			// 				else {
			// 					pl[nl].u = x;
			// 					pl[nl].v = y;
			// 					pl[nl].idepth = 0.01;
			// // 					pl[nl].idepth = 1;
			// 					//printf("the idepth is: %f\n", pl[nl].idepth);
			// 					pl[nl].iR = 0.01;
			// // 					pl[nl].iR = 1;
			// 					pl[nl].isGood=true;
			// 					pl[nl].energy.setZero();
			// 					pl[nl].lastHessian=0;
			// 					pl[nl].lastHessian_new=0;
			// 					pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
			// 					idepth[0][x+wl*y] = 0.01;
			// // 					idepth[0][x+wl*y] = 1;
			// 
			// 					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
			// 					float sumGrad2=0;
			// 					for(int idx=0;idx<patternNum;idx++)
			// 					{
			// 					    int dx = patternP[idx][0];
			// 					    int dy = patternP[idx][1];
			// 					    float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
			// 					    sumGrad2 += absgrad;
			// 					}
			// 
			// 					pl[nl].outlierTH = patternNum*setting_outlierTH;
			// 
			// 					nl++;
			// 					assert(nl <= npts);
			// 				}
			// 				delete pt;
			// 			}
			// 			if(lvl!=0 && statusMapB[x+y*wl])
			// 			{
			// 			  	int lvlm1 = lvl-1;
			// 				int wlm1 = w[lvlm1];
			// 				float* idepth_l = idepth[lvl];
			// 				float* idepth_lm = idepth[lvlm1];
			// 				//assert(patternNum==9);
			// 				pl[nl].u = x+0.1;
			// 				pl[nl].v = y+0.1;
			// 				pl[nl].idepth = 1;	
			// 				pl[nl].iR = 1;		
			// 				pl[nl].isGood=true;
			// 				pl[nl].energy.setZero();
			// 				pl[nl].lastHessian=0;
			// 				pl[nl].lastHessian_new=0;
			// 				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
			// 				int bidx = 2*x   + 2*y*wlm1;
			// 				idepth_l[x + y*wl] = idepth_lm[bidx] +
			// 											idepth_lm[bidx+1] +
			// 											idepth_lm[bidx+wlm1] +
			// 											idepth_lm[bidx+wlm1+1];
			// 
			// 				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
			// 				float sumGrad2=0;
			// 				for(int idx=0;idx<patternNum;idx++)
			// 				{
			// 					int dx = patternP[idx][0];
			// 					int dy = patternP[idx][1];
			// 					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
			// 					sumGrad2 += absgrad;
			// 				}
			// 
			// 				pl[nl].outlierTH = patternNum*setting_outlierTH;
			// 
			// 				nl++;
			// 				assert(nl <= npts);
			// 			}
			// 		}
			numPoints[lvl] = nl;
		}
		delete[] statusMap; statusMap = NULL;
		delete[] statusMapB; statusMapB = NULL;

		/*std::ofstream f2;
		std::string dsoposefile = "/home/sjm/桌面/temp/depth1_myself.txt";
		f2.open(dsoposefile, std::ios::out);
		for (int i = 0; i < numPoints[1]; i++)
			f2 << std::fixed << std::setprecision(9) << points[1][i].idepth << std::endl;
		f2.close();*/

		makeNN();

		thisToNext = SE3();
		snapped = false;
		frameID = snappedAt = 0;

		for (int i = 0; i < pyrLevelsUsed; i++)
			dGrads[i].setZero();
	}

	void CoarseInitializer::resetPoints(int lvl)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		for (int it = 0; it < npts; ++it)
		{
			pts[it].energy.setZero();
			pts[it].idepth_new = pts[it].idepth;

			// 使用邻近点的逆深度均值作为所遍历点的逆深度
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

			// 控制逆深度更新量的步进范围避免ZIGAZAG现象
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
			indexes[it] = NULL;
		}
	}
}