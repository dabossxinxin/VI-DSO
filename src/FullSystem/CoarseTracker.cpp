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
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/ImmaturePoint.h"

#include "IOWrapper/ImageRW.h"

#include "OptimizationBackend/EnergyFunctionalStructs.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
	/// <summary>
	/// 内存对齐函数
	/// </summary>
	/// <typeparam name="T">申请指针类型</typeparam>
	/// <param name="size">申请内存空间长度</param>
	/// <param name="rawPtrVec">内存空间原始地址</param>
	/// <returns>内存对齐后的地址</returns>
	template<int b, typename T>
	T* allocAligned(int size, std::vector<T*>& rawPtrVec)
	{
		const int padT = 1 + ((1 << b) / sizeof(T));
		T* ptr = new T[size + padT];
		rawPtrVec.emplace_back(ptr);
		T* alignedPtr = (T*)((((uintptr_t)(ptr + padT)) >> b) << b);
		return alignedPtr;
	}

	/// <summary>
	/// CoarseTracker构造函数，作用为分配成员变量的内存空间
	/// </summary>
	/// <param name="ww">跟踪图像的宽度</param>
	/// <param name="hh">跟踪图像的高度</param>
	CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0, 0)
	{
		// make coarse tracking templates.
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			int wl = ww >> lvl;
			int hl = hh >> lvl;

			idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

			pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
		}

		// warped buffers
		buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);

		newFrame = nullptr;
		lastRef = nullptr;
		debugPlot = true;
		debugPrint = true;
		w[0] = h[0] = 0;
		refFrameID = -1;
	}

	/// <summary>
	/// CoarseTracker析构函数，作用为清空内存空间
	/// </summary>
	CoarseTracker::~CoarseTracker()
	{
		for (auto ptr : ptrToDelete)
			SAFE_DELETE(ptr, true);
		ptrToDelete.clear();
	}

	/// <summary>
	/// 设置CoarseTracker的相机内参信息
	/// </summary>
	/// <param name="HCalib">相机内参信息</param>
	void CoarseTracker::makeK(CalibHessian* HCalib)
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
	/// 构造基于输入变量fh的深度图
	/// </summary>
	/// <param name="fh"></param>
	void CoarseTracker::makeCoarseDepthForFirstFrame(FrameHessian* fh)
	{
		// make coarse tracking templates for latstRef.
		memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
		memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

		for (PointHessian* ph : fh->pointHessians)
		{
			int u = ph->u + 0.5f;
			int v = ph->v + 0.5f;
			float new_idepth = ph->idepth;
			float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

			idepth[0][u + w[0] * v] += new_idepth * weight;
			weightSums[0][u + w[0] * v] += weight;
		}

		for (int lvl = 1; lvl < pyrLevelsUsed; ++lvl)
		{
			int lvlm1 = lvl - 1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			float* idepth_l = idepth[lvl];
			float* weightSums_l = weightSums[lvl];

			float* idepth_lm = idepth[lvlm1];
			float* weightSums_lm = weightSums[lvlm1];

			for (int y = 0; y < hl; ++y)
			{
				for (int x = 0; x < wl; ++x)
				{
					int bidx = 2 * x + 2 * y * wlm1;
					idepth_l[x + y * wl] = idepth_lm[bidx] +
						idepth_lm[bidx + 1] +
						idepth_lm[bidx + wlm1] +
						idepth_lm[bidx + wlm1 + 1];

					weightSums_l[x + y * wl] = weightSums_lm[bidx] +
						weightSums_lm[bidx + 1] +
						weightSums_lm[bidx + wlm1] +
						weightSums_lm[bidx + wlm1 + 1];
				}
			}
		}

		// dilate idepth by 1.
		for (int lvl = 0; lvl < 2; ++lvl)
		{
			int numIts = 1;

			for (int it = 0; it < numIts; ++it)
			{
				int wh = w[lvl] * h[lvl] - w[lvl];
				int wl = w[lvl];
				float* weightSumsl = weightSums[lvl];
				float* weightSumsl_bak = weightSums_bak[lvl];
				memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
				float* idepthl = idepth[lvl];	// dont need to make a temp copy of depth, since I only
												// read values with weightSumsl>0, and write ones with weightSumsl<=0.

				for (int i = w[lvl]; i < wh; ++i)
				{
					if (weightSumsl_bak[i] <= 0)
					{
						float sum = 0, num = 0, numn = 0;
						if (weightSumsl_bak[i + 1 + wl] > 0)
						{
							sum += idepthl[i + 1 + wl];
							num += weightSumsl_bak[i + 1 + wl];
							numn++;
						}
						if (weightSumsl_bak[i - 1 - wl] > 0)
						{
							sum += idepthl[i - 1 - wl];
							num += weightSumsl_bak[i - 1 - wl];
							numn++;
						}
						if (weightSumsl_bak[i + wl - 1] > 0)
						{
							sum += idepthl[i + wl - 1];
							num += weightSumsl_bak[i + wl - 1];
							numn++;
						}
						if (weightSumsl_bak[i - wl + 1] > 0)
						{
							sum += idepthl[i - wl + 1];
							num += weightSumsl_bak[i - wl + 1];
							numn++;
						}
						if (numn > 0)
						{
							idepthl[i] = sum / numn;
							weightSumsl[i] = num / numn;
						}
					}
				}
			}
		}

		// dilate idepth by 1 (2 on lower levels).
		for (int lvl = 2; lvl < pyrLevelsUsed; ++lvl)
		{
			int wh = w[lvl] * h[lvl] - w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
			float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.

			for (int i = w[lvl]; i < wh; ++i)
			{
				if (weightSumsl_bak[i] <= 0)
				{
					float sum = 0, num = 0, numn = 0;
					if (weightSumsl_bak[i + 1] > 0)
					{
						sum += idepthl[i + 1];
						num += weightSumsl_bak[i + 1];
						numn++;
					}
					if (weightSumsl_bak[i - 1] > 0)
					{
						sum += idepthl[i - 1];
						num += weightSumsl_bak[i - 1];
						numn++;
					}
					if (weightSumsl_bak[i + wl] > 0)
					{
						sum += idepthl[i + wl];
						num += weightSumsl_bak[i + wl];
						numn++;
					}
					if (weightSumsl_bak[i - wl] > 0)
					{
						sum += idepthl[i - wl];
						num += weightSumsl_bak[i - wl];
						numn++;
					}
					if (numn > 0)
					{
						idepthl[i] = sum / numn;
						weightSumsl[i] = num / numn;
					}
				}
			}
		}

		// normalize idepths and weights.
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			float* weightSumsl = weightSums[lvl];
			float* idepthl = idepth[lvl];
			Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

			int wl = w[lvl], hl = h[lvl];

			int lpc_n = 0;
			float* lpc_u = pc_u[lvl];
			float* lpc_v = pc_v[lvl];
			float* lpc_idepth = pc_idepth[lvl];
			float* lpc_color = pc_color[lvl];

			for (int y = 2; y < hl - 2; ++y)
			{
				for (int x = 2; x < wl - 2; ++x)
				{
					int i = x + y * wl;

					if (weightSumsl[i] > 0)
					{
						idepthl[i] /= weightSumsl[i];
						lpc_u[lpc_n] = x;
						lpc_v[lpc_n] = y;
						lpc_idepth[lpc_n] = idepthl[i];
						lpc_color[lpc_n] = dIRefl[i][0];

						if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
						{
							idepthl[i] = -1;
							continue;	// just skip if something is wrong.
						}
						lpc_n++;
					}
					else
						idepthl[i] = -1;

					weightSumsl[i] = 1;
				}
			}

			pc_n[lvl] = lpc_n;
		}
	}

	/// <summary>
	/// 使用滑窗中所有关键帧构造基于成员变量lastRef的深度图
	/// </summary>
	/// <param name="frameHessians">滑窗中所有关键帧</param>
	/// <param name="fh_right"></param>
	/// <param name="Hcalib">相机的内参</param>
	void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fhRight, CalibHessian Hcalib)
	{
		memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
		memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

		for (FrameHessian* fh : frameHessians)
		{
			for (PointHessian* ph : fh->pointHessians)
			{
				if (ph->lastResiduals[0].first != nullptr && ph->lastResiduals[0].second == ResState::INNER)
				{
					PointFrameResidual* r = ph->lastResiduals[0].first;
					// TODO:如何保证特征点最近计算的残差关键帧就是tracker中的参考帧
					assert(r->efResidual->isActive() && r->target == lastRef);

					// 特征在lastRef中的投影像素点
					int u = r->centerProjectedTo[0] + 0.5f;
					int v = r->centerProjectedTo[1] + 0.5f;
					float new_idepth = r->centerProjectedTo[2];
					float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

					idepth[0][u + w[0] * v] += new_idepth * weight;
					weightSums[0][u + w[0] * v] += weight;
				}
			}
		}

		// 将上一步得到的逆深度向高层金字塔传播
		for (int lvl = 1; lvl < pyrLevelsUsed; ++lvl)
		{
			int lvlm1 = lvl - 1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			float* idepth_l = idepth[lvl];
			float* weightSums_l = weightSums[lvl];

			float* idepth_lm = idepth[lvlm1];
			float* weightSums_lm = weightSums[lvlm1];

			for (int y = 0; y < hl; ++y)
			{
				for (int x = 0; x < wl; ++x)
				{
					int bidx = 2 * x + 2 * y * wlm1;
					idepth_l[x + y * wl] = idepth_lm[bidx] +
						idepth_lm[bidx + 1] +
						idepth_lm[bidx + wlm1] +
						idepth_lm[bidx + wlm1 + 1];

					weightSums_l[x + y * wl] = weightSums_lm[bidx] +
						weightSums_lm[bidx + 1] +
						weightSums_lm[bidx + wlm1] +
						weightSums_lm[bidx + wlm1 + 1];
				}
			}
		}

		// 将逆深度向图像周围几个像素膨胀一定次数：
		// 低层级金字塔特征相对稀疏，膨胀次数设大一点；
		// 高层及金字塔特征相对密集，膨胀次数设小一点
		for (int lvl = 0; lvl < 2; ++lvl)
		{
			int numIts = 1;

			for (int it = 0; it < numIts; ++it)
			{
				int wh = w[lvl] * h[lvl] - w[lvl];
				int wl = w[lvl];
				float* weightSumsl = weightSums[lvl];
				float* weightSumsl_bak = weightSums_bak[lvl];
				memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
				float* idepthl = idepth[lvl];

				// 为了保证循环体内不越界，for循环去掉了第一行和最后一行
				for (int i = w[lvl]; i < wh; ++i)
				{
					if (weightSumsl_bak[i] <= 0)
					{
						float sum = 0, num = 0, numn = 0;
						if (weightSumsl_bak[i + 1 + wl] > 0)
						{
							sum += idepthl[i + 1 + wl];
							num += weightSumsl_bak[i + 1 + wl];
							numn++;
						}
						if (weightSumsl_bak[i - 1 - wl] > 0)
						{
							sum += idepthl[i - 1 - wl];
							num += weightSumsl_bak[i - 1 - wl];
							numn++;
						}
						if (weightSumsl_bak[i + wl - 1] > 0)
						{
							sum += idepthl[i + wl - 1];
							num += weightSumsl_bak[i + wl - 1];
							numn++;
						}
						if (weightSumsl_bak[i - wl + 1] > 0)
						{
							sum += idepthl[i - wl + 1];
							num += weightSumsl_bak[i - wl + 1];
							numn++;
						}
						if (numn > 0)
						{
							idepthl[i] = sum / numn;
							weightSumsl[i] = num / numn;
						}
					}
				}
			}
		}

		// 高金字塔层级的逆深度向周围像素传播
		for (int lvl = 2; lvl < pyrLevelsUsed; ++lvl)
		{
			int wh = w[lvl] * h[lvl] - w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
			float* idepthl = idepth[lvl];

			// 为了保证循环体内不越界，for循环去掉了图像第一行和最后一行
			for (int i = w[lvl]; i < wh; ++i)
			{
				if (weightSumsl_bak[i] <= 0)
				{
					float sum = 0, num = 0, numn = 0;
					if (weightSumsl_bak[i + 1] > 0)
					{
						sum += idepthl[i + 1];
						num += weightSumsl_bak[i + 1];
						numn++;
					}
					if (weightSumsl_bak[i - 1] > 0)
					{
						sum += idepthl[i - 1];
						num += weightSumsl_bak[i - 1];
						numn++;
					}
					if (weightSumsl_bak[i + wl] > 0)
					{
						sum += idepthl[i + wl];
						num += weightSumsl_bak[i + wl];
						numn++;
					}
					if (weightSumsl_bak[i - wl] > 0)
					{
						sum += idepthl[i - wl];
						num += weightSumsl_bak[i - wl];
						numn++;
					}
					if (numn > 0)
					{
						idepthl[i] = sum / numn;
						weightSumsl[i] = num / numn;
					}
				}
			}
		}

		// 归一化逆深度
		for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
		{
			float* weightSumsl = weightSums[lvl];
			float* idepthl = idepth[lvl];
			Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

			int wl = w[lvl], hl = h[lvl];

			int lpc_n = 0;
			float* lpc_u = pc_u[lvl];
			float* lpc_v = pc_v[lvl];
			float* lpc_idepth = pc_idepth[lvl];
			float* lpc_color = pc_color[lvl];

			// 边缘像素不可信，遍历的时候直接去掉
			for (int y = 2; y < hl - 2; ++y)
			{
				for (int x = 2; x < wl - 2; ++x)
				{
					int i = x + y * wl;

					if (weightSumsl[i] > 0)
					{
						idepthl[i] /= weightSumsl[i];
						lpc_u[lpc_n] = x;
						lpc_v[lpc_n] = y;
						lpc_idepth[lpc_n] = idepthl[i];
						lpc_color[lpc_n] = dIRefl[i][0];

						if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
						{
							idepthl[i] = -1;
							continue;
						}
						lpc_n++;
					}
					else
						idepthl[i] = -1;

					weightSumsl[i] = 1;
				}
			}

			pc_n[lvl] = lpc_n;
		}
	}

	/// <summary>
	/// 计算视觉部分Heesian和b
	/// </summary>
	/// <param name="lvl">输入金字塔层级</param>
	/// <param name="H_out">输出当前层金字塔计算的H</param>
	/// <param name="b_out">输出当前层金字塔计算的b</param>
	/// <param name="refToNew">输入位姿变换参数</param>
	/// <param name="aff_g2l">输入光度变换参数</param>
	void CoarseTracker::calcGSSSE(int lvl, Mat88& H_out, Vec8& b_out, const SE3& refToNew, AffLight aff_g2l)
	{
		acc.initialize();

		__m128 fxl = _mm_set1_ps(fx[lvl]);
		__m128 fyl = _mm_set1_ps(fy[lvl]);
		__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
		__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

		__m128 one = _mm_set1_ps(1);
		__m128 minusOne = _mm_set1_ps(-1);
		__m128 zero = _mm_set1_ps(0);

		assert(buf_warped_n % 4 == 0);
		for (int it = 0; it < buf_warped_n; it += 4)
		{
			__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + it), fxl);
			__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + it), fyl);
			__m128 u = _mm_load_ps(buf_warped_u + it);
			__m128 v = _mm_load_ps(buf_warped_v + it);
			__m128 id = _mm_load_ps(buf_warped_idepth + it);

			acc.updateSSE_weighted(
				_mm_mul_ps(id, dx),																		// dres/dtran0
				_mm_mul_ps(id, dy),																		// dres/dtran1
				_mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),		// dres/dtran2
				_mm_sub_ps(zero, _mm_add_ps(
					_mm_mul_ps(_mm_mul_ps(u, v), dx),
					_mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),								// dres/drot0
				_mm_add_ps(
					_mm_mul_ps(_mm_mul_ps(u, v), dy),
					_mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),									// dres/drot1
				_mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),										// dres/drot2
				_mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + it))),					// dres/da
				minusOne,																				// dres/db
				_mm_load_ps(buf_warped_residual + it),													// dres
				_mm_load_ps(buf_warped_weight + it));													// huber weight
		}

		acc.finish();
		double buf_warped_n_fac = 1.0 / buf_warped_n;
		H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * buf_warped_n_fac;
		b_out = acc.H.topRightCorner<8, 1>().cast<double>() * buf_warped_n_fac;

		// 为了平衡H矩阵各部分的值，这里给每个部分乘上不同的尺度因子
		H_out.block<8, 3>(0, 0) *= SCALE_XI_TRANS;
		H_out.block<8, 3>(0, 3) *= SCALE_XI_ROT;
		H_out.block<8, 1>(0, 6) *= SCALE_A;
		H_out.block<8, 1>(0, 7) *= SCALE_B;
		H_out.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
		H_out.block<3, 8>(3, 0) *= SCALE_XI_ROT;
		H_out.block<1, 8>(6, 0) *= SCALE_A;
		H_out.block<1, 8>(7, 0) *= SCALE_B;
		b_out.segment<3>(0) *= SCALE_XI_TRANS;
		b_out.segment<3>(3) *= SCALE_XI_ROT;
		b_out.segment<1>(6) *= SCALE_A;
		b_out.segment<1>(7) *= SCALE_B;
	}

	/// <summary>
	/// 计算当前输入状态下的视觉残差以及特征在最新帧上的信息
	/// </summary>
	/// <param name="lvl">计算视觉残差的金字塔层级</param>
	/// <param name="refToNew">输入参考帧到最新帧的位姿变换</param>
	/// <param name="aff_g2l">输入光度变换参数</param>
	/// <param name="cutoffTH">输入单个特征光度误差阈值</param>
	/// <returns>
	/// [0]：总的光度残差能量；
	/// [1]：统计能量的特征数量；
	/// [2]：仅考虑平移时特征在像素坐标下的移动距离；
	/// [3]：固定为0；
	/// [4]：考虑平移和旋转时特征在像素坐标下的移动距离；
	/// [5]：计算总的光度残差时大于设定阈值的特征比例；
	/// </returns>
	Vec6 CoarseTracker::calcRes(int lvl, const SE3& refToNew, AffLight aff_g2l, float cutoffTH)
	{
		float E = 0;
		int numTermsInE = 0;		// 参与统计光度残差的点的个数
		int numTermsInWarped = 0;	// 参与统计光度残差点中残差满足阈值的点个数
		int numSaturated = 0;		// 参与统计光度残差点中残差大于阈值的点个数

		int wl = w[lvl];
		int hl = h[lvl];
		Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
		float fxl = fx[lvl];
		float fyl = fy[lvl];
		float cxl = cx[lvl];
		float cyl = cy[lvl];

		Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
		Vec3f t = (refToNew.translation()).cast<float>();
		Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();

		float sumSquaredShiftT = 0;
		float sumSquaredShiftRT = 0;
		float sumSquaredShiftNum = 0;

		float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH * setting_huberTH;	// energy for r=setting_coarseCutoffTH.

		MinimalImageB3* resImage = nullptr;
		if (debugPlot)
		{
			resImage = new MinimalImageB3(wl, hl);
			resImage->setConst(Vec3b(255, 255, 255));
		}

		int nl = pc_n[lvl];
		float* lpc_u = pc_u[lvl];
		float* lpc_v = pc_v[lvl];
		float* lpc_idepth = pc_idepth[lvl];
		float* lpc_color = pc_color[lvl];

		for (int i = 0; i < nl; i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];

			Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = id / pt[2];

			// 在第0层金字塔中计算光流信息
			if (lvl == 0 && i % 32 == 0)
			{
				// translation only (positive)
				Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				// translation only (negative)
				Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.
				sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
				sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
				sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
				sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
				sumSquaredShiftNum += 2;
			}

			if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) continue;

			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
			if (!std::isfinite((float)hitColor[0])) continue;
			float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			if (fabs(residual) > cutoffTH)
			{
				if (debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
				E += maxEnergy;
				numTermsInE++;
				numSaturated++;
			}
			else
			{
				if (debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual + 128, residual + 128, residual + 128));

				E += hw * residual * residual * (2 - hw);
				numTermsInE++;

				buf_warped_idepth[numTermsInWarped] = new_idepth;
				buf_warped_u[numTermsInWarped] = u;
				buf_warped_v[numTermsInWarped] = v;
				buf_warped_dx[numTermsInWarped] = hitColor[1];
				buf_warped_dy[numTermsInWarped] = hitColor[2];
				buf_warped_residual[numTermsInWarped] = residual;
				buf_warped_weight[numTermsInWarped] = hw;
				buf_warped_refColor[numTermsInWarped] = lpc_color[i];
				numTermsInWarped++;
			}
		}

		// 为保证使用SSE计算时的性能将多余的数据去除
		while (numTermsInWarped % 4 != 0)
		{
			buf_warped_idepth[numTermsInWarped] = 0;
			buf_warped_u[numTermsInWarped] = 0;
			buf_warped_v[numTermsInWarped] = 0;
			buf_warped_dx[numTermsInWarped] = 0;
			buf_warped_dy[numTermsInWarped] = 0;
			buf_warped_residual[numTermsInWarped] = 0;
			buf_warped_weight[numTermsInWarped] = 0;
			buf_warped_refColor[numTermsInWarped] = 0;
			numTermsInWarped++;
		}
		buf_warped_n = numTermsInWarped;

		if (debugPlot)
		{
			IOWrap::displayImage("RES", resImage, false);
			IOWrap::waitKey(0);
			SAFE_DELETE(resImage);
		}

		Vec6 rs;
		rs[0] = E;
		rs[1] = numTermsInE;
		rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
		rs[3] = 0;
		rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
		rs[5] = numSaturated / (float)numTermsInE;
		return rs;
	}

	/// <summary>
	/// 计算惯导预积分数据的Hessian和b
	/// </summary>
	/// <param name="H_out">输出惯导数据H</param>
	/// <param name="b_out">输出惯导数据b</param>
	/// <param name="refToNew">输入位姿变换参数</param>
	/// <param name="IMU_preintegrator">IMU预积分handle</param>
	/// <param name="res_PVPhi">IMU预积分数据位姿及速度残差</param>
	/// <param name="PointEnergy">这个函数中好像并没有用到</param>
	/// <param name="imu_track_weight">IMU信息的权重</param>
	/// <returns>IMU信息残差：包含位置残差和姿态残差</returns>
	double CoarseTracker::calcIMUResAndGS(Mat66& H_out, Vec6& b_out, SE3& refToNew, const IMUPreintegrator& IMU_preintegrator, Vec9& res_PVPhi, double PointEnergy, double trackWeight)
	{
		Mat44 M_WD = T_WD.matrix();
		SE3 newToRef = refToNew.inverse();

		// 跟踪模块参考帧的位姿
		Mat44 M_DC_i = lastRef->shell->camToWorld.matrix();
		SE3 T_WB_i(M_WD * M_DC_i * M_WD.inverse() * T_BC.inverse().matrix());
		Mat33 R_WB_i = T_WB_i.rotationMatrix();
		Vec3 t_WB_i = T_WB_i.translation();

		// 跟踪模块最新帧的位姿
		Mat44 M_DC_j = (lastRef->shell->camToWorld * newToRef).matrix();
		SE3 T_WB_j(M_WD * M_DC_j * M_WD.inverse() * T_BC.inverse().matrix());
		Mat33 R_WB_j = T_WB_j.rotationMatrix();
		Vec3 t_WB_j = T_WB_j.translation();

		double dt = IMU_preintegrator.getDeltaTime();

		H_out = Mat66::Zero();
		b_out = Vec6::Zero();
		if (dt > 0.5)
		{
			printf("WARNING: imu integration dt > 0.5s in tracking module!\n");
			return 0;
		}

		Vec3 g_w(0, 0, -setting_gravityNorm);

		// 计算p、v、q残差
		Mat33 R_temp = SO3::exp(IMU_preintegrator.getJRBiasg() * lastRef->delta_bias_g).matrix();
		Vec3 res_phi = SO3((IMU_preintegrator.getDeltaR() * R_temp).transpose() * R_WB_i.transpose() * R_WB_j).log();

		Vec3 delta_v = IMU_preintegrator.getDeltaV() + IMU_preintegrator.getJVBiasa() * lastRef->delta_bias_a +
			IMU_preintegrator.getJVBiasg() * lastRef->delta_bias_g;
		newFrame->velocity = R_WB_i * (R_WB_i.transpose() * (lastRef->velocity + g_w * dt) + delta_v);
		Vec3 res_v = R_WB_i.transpose() * (newFrame->velocity - lastRef->velocity - g_w * dt) - delta_v;
		newFrame->shell->velocity = newFrame->velocity;

		Vec3 delta_p = IMU_preintegrator.getDeltaP() + IMU_preintegrator.getJPBiasa() * lastRef->delta_bias_a +
			IMU_preintegrator.getJPBiasg() * lastRef->delta_bias_g;
		Vec3 res_p = R_WB_i.transpose() * (t_WB_j - t_WB_i - lastRef->velocity * dt - 0.5 * g_w * dt * dt) - delta_p;

		Mat99 Cov = IMU_preintegrator.getCovPVPhi();

		res_PVPhi.block(0, 0, 3, 1) = res_p;
		res_PVPhi.block(3, 0, 3, 1) = Vec3::Zero();
		res_PVPhi.block(6, 0, 3, 1) = res_phi;

		double res = trackWeight * trackWeight * res_PVPhi.transpose() * Cov.inverse() * res_PVPhi;

		Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi);
		Mat33 J_resV_v_j = R_WB_i.transpose();
		Mat33 J_resP_p_j = R_WB_i.transpose() * R_WB_j;

		Mat66 J_imu_j = Mat66::Zero();
		J_imu_j.block(0, 0, 3, 3) = J_resP_p_j;
		J_imu_j.block(3, 3, 3, 3) = J_resPhi_phi_j;
		//J_imu_tmp.block(6,6,3,3) = J_resV_v_j;

		Mat66 Weight = Mat66::Zero();
		Weight.block(0, 0, 3, 3) = Cov.block(0, 0, 3, 3);
		Weight.block(3, 3, 3, 3) = Cov.block(6, 6, 3, 3);
		//Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
		Weight = Weight.diagonal().asDiagonal().inverse();
		Weight *= (trackWeight * trackWeight);

		Vec6 r_imu = Vec6::Zero();
		r_imu.block(0, 0, 3, 1) = res_p;
		r_imu.block(3, 0, 3, 1) = res_phi;
		//r_imu.block(6,0,3,1) = res_v;

		Mat44 T_tmp = T_BC.matrix() * T_WD.matrix() * M_DC_j.inverse();
		Mat66 J_rel = (-1 * Sim3(T_tmp).Adj()).block(0, 0, 6, 6);
		Mat66 J_xi_2_th = SE3(M_DC_i).Adj();				// 绝对位姿对相对位姿的雅可比

		Mat66 J_xi_r_l = refToNew.Adj().inverse();			// 将右扰动转化为左扰动
		Mat66 J_imu = Mat66::Zero();
		J_imu = J_imu_j * J_rel * J_xi_2_th * J_xi_r_l;

		H_out = J_imu.transpose() * Weight * J_imu;
		b_out = J_imu.transpose() * Weight * r_imu;

		H_out.block<6, 3>(0, 0) *= SCALE_XI_TRANS;
		H_out.block<6, 3>(0, 3) *= SCALE_XI_ROT;
		H_out.block<3, 6>(0, 0) *= SCALE_XI_TRANS;
		H_out.block<3, 6>(3, 0) *= SCALE_XI_ROT;
		//H_out.block<9,3>(0,6) *= SCALE_V;
		//H_out.block<3,9>(6,0) *= SCALE_V;

		b_out.segment<3>(0) *= SCALE_XI_TRANS;
		b_out.segment<3>(3) *= SCALE_XI_ROT;
		//b_out.segment<3>(6) *= SCALE_V;

		return res;
	}

	/// <summary>
	/// 设置跟踪的参考帧：将滑窗中的最后一帧关键帧设置为参考帧
	/// </summary>
	/// <param name="frameHessians">滑窗中所有关键帧</param>
	void CoarseTracker::setCTRefForFirstFrame(std::vector<FrameHessian*> frameHessians)
	{
		assert(frameHessians.size() > 0);
		lastRef = frameHessians.back();

		makeCoarseDepthForFirstFrame(lastRef);

		refFrameID = lastRef->shell->id;
		lastRef_aff_g2l = lastRef->aff_g2l();

		firstCoarseRMSE = -1;
	}

	/// <summary>
	/// 设置跟踪的参考帧：将滑窗中的最后一帧关键帧设置为参考帧
	/// </summary>
	/// <param name="frameHessians">滑窗中所有的关键帧</param>
	/// <param name="fh_right">在函数中好像并没有用到</param>
	/// <param name="Hcalib">相机内参信息</param>
	void CoarseTracker::setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians, FrameHessian* fhRight, CalibHessian Hcalib)
	{
		assert(frameHessians.size() > 0);
		lastRef = frameHessians.back();
		makeCoarseDepthL0(frameHessians, fhRight, Hcalib);

		refFrameID = lastRef->shell->id;
		lastRef_aff_g2l = lastRef->aff_g2l();

		firstCoarseRMSE = -1;
	}

	bool CoarseTracker::trackNewestCoarse(FrameHessian* newFrameHessian, SE3& lastToNew_out, AffLight& aff_g2l_out,
		int coarsestLvl, Vec5 minResForAbort, IOWrap::Output3DWrapper* wrap)
	{
		debugPlot = setting_render_displayCoarseTrackingFull;
		debugPrint = false;

		assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

		lastResiduals.setConstant(NAN);
		lastFlowIndicators.setConstant(1000);

		newFrame = newFrameHessian;
		int maxIterations[] = { 10,20,50,50,50 };
		float lambdaExtrapolationLimit = 0.001;

		SE3 refToNew_current = lastToNew_out;
		AffLight aff_g2l_current = aff_g2l_out;

		bool haveRepeated = false;

		// 计算惯导对应的Hessian信息
		IMUPreintegrator IMU_preintegrator;
		IMU_preintegrator.reset();
		double timeStart = input_picTimestampLeftList[lastRef->shell->incomingId];
		double timeEnd = input_picTimestampLeftList[newFrame->shell->incomingId];
		preintergrate(IMU_preintegrator, lastRef->bias_g, lastRef->bias_a, timeStart, timeEnd);

		// TODO：为何随着金字塔层级增加，IMU数据的权重逐渐降低
		std::vector<double> imuTrackWeight(coarsestLvl + 1, 0);
		imuTrackWeight[0] = setting_imuWeightTracker;
		imuTrackWeight[1] = imuTrackWeight[0] / 1.2;
		imuTrackWeight[2] = imuTrackWeight[1] / 1.5;
		imuTrackWeight[3] = imuTrackWeight[2] / 2;
		imuTrackWeight[4] = imuTrackWeight[3] / 3;

		// 自顶层向金字塔底层的视觉跟踪
		for (int lvl = coarsestLvl; lvl >= 0; lvl--)
		{
			// 1、计算视觉部分的Hessian
			Mat88 H_visual;
			Vec8 b_visual;
			float levelCutoffRepeat = 1;
			Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
			while (resOld[5] > 0.6 && levelCutoffRepeat < 50)
			{
				levelCutoffRepeat *= 2;
				resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

				if (!setting_debugout_runquiet)
					printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
			}
			calcGSSSE(lvl, H_visual, b_visual, refToNew_current, aff_g2l_current);

			// 2、计算惯导部分的Hessian
			double resImuOld = 0;
			Mat66 H_imu; 
			Vec6 b_imu;
			Vec9 res_PVPhi;
			if (lvl == 0) resImuOld = calcIMUResAndGS(H_imu, b_imu, refToNew_current, IMU_preintegrator, res_PVPhi, resOld[0], imuTrackWeight[lvl]);

			float lambda = 0.01;

			if (debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
				printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					0, (int)resOld[1],
					0.0f);
				std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
			}

			// 3、迭代优化计算跟踪结果
			for (int iteration = 0; iteration < maxIterations[lvl]; iteration++)
			{
				Vec8 bl = b_visual;
				Mat88 Hl = H_visual;
				if (setting_useImu && setting_imuTrackFlag && setting_imuTrackReady && lvl == 0)
				{
					Hl.block(0, 0, 6, 6) = H_visual.block(0, 0, 6, 6) + H_imu;
					bl.block(0, 0, 6, 1) = b_visual.block(0, 0, 6, 1) + b_imu;
				}

				for (int it = 0; it < 8; ++it) Hl(it, it) *= (1 + lambda);
				Vec8 inc = Hl.ldlt().solve(-bl);

				// 固定光度参数a21以及b21不进行优化
				if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)
				{
					inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-bl.head<6>());
					inc.tail<2>().setZero();
				}
				// 固定光度参数b21但是参数a21不固定
				if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
				{
					inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-bl.head<7>());
					inc.tail<1>().setZero();
				}
				// 固定光度参数a21但是参数b21不固定
				if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))
				{
					Mat88 HlStitch = Hl;
					Vec8 bStitch = bl;
					HlStitch.col(6) = HlStitch.col(7);
					HlStitch.row(6) = HlStitch.row(7);
					bStitch[6] = bStitch[7];
					Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
					inc.setZero();
					inc.head<6>() = incStitch.head<6>();
					inc[6] = 0;
					inc[7] = incStitch[6];
				}

				float extrapFac = 1;
				if (lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
				inc *= extrapFac;

				// 求解时为了平衡Hessian中各部分的数值对Hessian进行了尺度处理，得到结果后需要对结果也进行相同处理
				Vec8 incScaled = inc;
				incScaled.segment<3>(0) *= SCALE_XI_TRANS;
				incScaled.segment<3>(3) *= SCALE_XI_ROT;
				incScaled.segment<1>(6) *= SCALE_A;
				incScaled.segment<1>(7) *= SCALE_B;

				if (!std::isfinite(incScaled.sum())) incScaled.setZero();

				// 求导的时候用的是左扰动，因此将增量赋给结果时应该左乘
				SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
				AffLight aff_g2l_new = aff_g2l_current;
				aff_g2l_new.a += incScaled[6];
				aff_g2l_new.b += incScaled[7];

				Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);
				double resImuNew = 0;
				if (lvl == 0) resImuNew = calcIMUResAndGS(H_imu, b_imu, refToNew_new, IMU_preintegrator, res_PVPhi, resNew[0], imuTrackWeight[lvl]);

				// 计算能量值是否降低来判断此次迭代过程是否被接受
				bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);
				if (setting_useImu && setting_imuTrackFlag && setting_imuTrackReady && lvl == 0)
					accept = (resNew[0] / resNew[1] + resImuNew) < (resOld[0] / resOld[1] + resImuOld);

				if (debugPrint)
				{
					Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
					printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
					std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
				}

				if (accept)
				{
					calcGSSSE(lvl, H_visual, b_visual, refToNew_new, aff_g2l_new);
					resOld = resNew;
					resImuOld = resImuNew;
					aff_g2l_current = aff_g2l_new;
					refToNew_current = refToNew_new;
					lambda *= 0.5;
				}
				else
				{
					lambda *= 4;
					if (lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
				}

				if (!(inc.norm() > 1e-3))
				{
					if (debugPrint)
						printf("inc too small, break!\n"); 
					break;
				}
			}

			// set last residual for that level, as well as flow indicators.
			lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
			lastFlowIndicators = resOld.segment<3>(2);
			if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl]) return false;

			// 单个特征光度误差阈值放大了，那么这一层金字塔需要再按照流程计算一遍
			if (levelCutoffRepeat > 1 && !haveRepeated)
			{
				lvl++;
				haveRepeated = true;
				printf("REPEAT LEVEL!\n");
			}
		}

		lastToNew_out = refToNew_current;
		aff_g2l_out = aff_g2l_current;

		// 4、检查得到的参数是否异常，若存在异常直接返回跟踪失败
		if ((setting_affineOptModeA != 0 && (std::abs(aff_g2l_out.a) > 1.2))
			|| (setting_affineOptModeB != 0 && (std::abs(aff_g2l_out.b) > 200)))
		{
			printf("affine parameter error: %.3f, %.3f\n",
				std::abs(aff_g2l_out.a), std::abs(aff_g2l_out.b));
			return false;
		}

		Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

		if ((setting_affineOptModeA == 0 && (std::abs(logf((float)relAff[0])) > 1.5))
			|| (setting_affineOptModeB == 0 && (std::abs((float)relAff[1]) > 200)))
		{
			printf("relative affine parameter error: %.3f, %.3f\n",
				std::abs(logf((float)relAff[0])), std::abs((float)relAff[1]));
			return false;
		}

		if (setting_affineOptModeA < 0) aff_g2l_out.a = 0;
		if (setting_affineOptModeB < 0) aff_g2l_out.b = 0;

		return true;
	}

	void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*>& wraps)
	{
		if (w[1] == 0) return;
		int lvl = 0;

		{
			int wl = w[lvl];
			int hl = h[lvl];
			int wh = wl * hl;
			std::vector<float> allID;
			
			// 拿到滑窗关键帧中的逆深度数据并排序，拿出最大和最小的逆深度数据
			for (int it = 0; it < wh; ++it)
			{
				if (idepth[lvl][it] > 0)
					allID.emplace_back(idepth[lvl][it]);
			}
			std::sort(allID.begin(), allID.end(), [&](float a, float b) {return a < b; });
			int n = allID.size() - 1;

			if (allID.empty())
			{
				printf("please check feature param, all idepth vector empty.\n");
				return;
			}

			float minID_new = allID[(int)(n * 0.05)];
			float maxID_new = allID[(int)(n * 0.95)];

			float minID, maxID;
			minID = minID_new;
			maxID = maxID_new;
			if (minID_pt != 0 && maxID_pt != 0)
			{
				if (*minID_pt < 0 || *maxID_pt < 0)
				{
					*maxID_pt = maxID;
					*minID_pt = minID;
				}
				else
				{
					// slowly adapt: change by maximum 10% of old span.
					float maxChange = 0.3 * (*maxID_pt - *minID_pt);

					if (minID < *minID_pt - maxChange)
						minID = *minID_pt - maxChange;
					if (minID > *minID_pt + maxChange)
						minID = *minID_pt + maxChange;

					if (maxID < *maxID_pt - maxChange)
						maxID = *maxID_pt - maxChange;
					if (maxID > *maxID_pt + maxChange)
						maxID = *maxID_pt + maxChange;

					*maxID_pt = maxID;
					*minID_pt = minID;
				}
			}

			// 绘制跟踪模块中参考关键帧的图像数据以及特征逆深度数据，即深度图
			MinimalImageB3 mf(wl, hl);
			mf.setBlack();
			int color = 0;
			for (int it = 0; it < wh; ++it)
			{
				color = lastRef->dIp[lvl][it][0] * 0.9f;
				if (color > 255) color = 255;
				mf.at(it) = Vec3b(color, color, color);
			}

			for (int y = 3; y < hl - 3; ++y)
			{
				for (int x = 3; x < wl - 3; ++x)
				{
					int idx = x + y * wl;
					float sid = 0, nid = 0;
					float* bp = idepth[lvl] + idx;

					if (bp[0] > 0) { sid += bp[0]; nid++; }
					if (bp[1] > 0) { sid += bp[1]; nid++; }
					if (bp[-1] > 0) { sid += bp[-1]; nid++; }
					if (bp[wl] > 0) { sid += bp[wl]; nid++; }
					if (bp[-wl] > 0) { sid += bp[-wl]; nid++; }

					if (bp[0] > 0 || nid >= 3)
					{
						float id = ((sid / nid) - minID) / ((maxID - minID));
						mf.setPixelCirc(x, y, makeJet3B(id));
					}
				}
			}

			// 将深度图数据加入到显示线程中显示，并把这个数据保存在磁盘中
			for (IOWrap::Output3DWrapper* ow : wraps)
				ow->pushDepthImage(&mf);

			if (setting_debugSaveImages)
			{
				char buf[1000];
				snprintf(buf, 1000, "images_out/tracked_ref_%05d_%05d.png", lastRef->shell->id, refFrameID);
				IOWrap::writeImage(buf, &mf);
			}
		}
	}

	/// <summary>
	/// 跟踪调试时输出参考帧逆深度的热力图
	/// </summary>
	/// <param name="wraps">pangolin显示窗口</param>
	void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*>& wraps)
	{
		if (w[1] == 0) return;
		int lvl = 0;
		MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
		for (IOWrap::Output3DWrapper* ow : wraps)
			ow->pushDepthImageFloat(&mim, lastRef);
	}

	/***************************CoarseDistanceMap***************************/
	CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
	{
		fwdWarpedIDDistFinal = new unsigned char[ww * hh / 4];

		bfsList1 = new Eigen::Vector2i[ww * hh / 4];
		bfsList2 = new Eigen::Vector2i[ww * hh / 4];

		int fac = 1 << (pyrLevelsUsed - 1);

		coarseProjectionGrid = new PointFrameResidual * [2048 * (ww * hh / (fac * fac))];
		coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

		w[0] = h[0] = 0;
		debugImage = nullptr;
	}

	CoarseDistanceMap::~CoarseDistanceMap()
	{
		SAFE_DELETE(debugImage);
		SAFE_DELETE(fwdWarpedIDDistFinal, true);
		SAFE_DELETE(bfsList1, true);
		SAFE_DELETE(bfsList2, true);
		SAFE_DELETE(coarseProjectionGrid, true);
		SAFE_DELETE(coarseProjectionGridNum, true);
	}

	void CoarseDistanceMap::makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian* frame)
	{
		int w1 = w[1];
		int h1 = h[1];
		int wh1 = w1 * h1;
		int numItems = 0;

		memset(fwdWarpedIDDistFinal, 255, sizeof(unsigned char) * wh1);

		// 滑窗关键帧中管理的特征都投影到最新关键帧frame中
		for (FrameHessian* fh : frameHessians)
		{
			if (frame == fh) continue;

			SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
			Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
			Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

			// 将激活的特征加入到距离地图中
			for (PointHessian* ph : fh->pointHessians)
			{
				assert(ph->status == PointHessian::ACTIVE);

				// 计算关键帧特征在最新帧上的投影像素坐标
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				//过滤掉边缘越界的像素坐标
				if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;

				fwdWarpedIDDistFinal[u + w1 * v] = 0;
				bfsList1[numItems] = Eigen::Vector2i(u, v);
				numItems++;
			}
		}

		growDistBFS(numItems);
	}

	void CoarseDistanceMap::debugPlotDistanceMap()
	{
		int wl = w[1];
		int hl = h[1];
		int whl = wl * hl;

		debugImage = new MinimalImageB(wl, hl);
		memcpy(debugImage->data, fwdWarpedIDDistFinal, sizeof(unsigned char) * whl);
	}

	void CoarseDistanceMap::growDistBFS(int bfsNum)
	{
		assert(w[0] != 0);
		int w1 = w[1], h1 = h[1];
		Eigen::Vector2i* bfsListTmp;

		for (int k = 1; k < 40; ++k)
		{
			int bfsNum2 = bfsNum;

			bfsListTmp = bfsList1;
			bfsList1 = bfsList2;
			bfsList2 = bfsListTmp;

			bfsNum = 0;

			if (k % 2 == 0)
			{
				for (int i = 0; i < bfsNum2; ++i)
				{
					int x = bfsList2[i][0];
					int y = bfsList2[i][1];
					if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
					int idx = x + y * w1;

					// 上、下、左、右
					if (fwdWarpedIDDistFinal[idx + 1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x + 1, y);
					}
					if (fwdWarpedIDDistFinal[idx - 1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x - 1, y);
					}
					if (fwdWarpedIDDistFinal[idx + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x, y + 1);
					}
					if (fwdWarpedIDDistFinal[idx - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x, y - 1);
					}
				}
			}
			else
			{
				for (int i = 0; i < bfsNum2; ++i)
				{
					int x = bfsList2[i][0];
					int y = bfsList2[i][1];
					if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
					int idx = x + y * w1;

					// 上、下、左、右、左上、右上、左下、右下
					if (fwdWarpedIDDistFinal[idx + 1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x + 1, y);
					}
					if (fwdWarpedIDDistFinal[idx - 1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x - 1, y);
					}
					if (fwdWarpedIDDistFinal[idx + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x, y + 1);
					}
					if (fwdWarpedIDDistFinal[idx - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x, y - 1);
					}

					if (fwdWarpedIDDistFinal[idx + 1 + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1 + w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x + 1, y + 1);
					}
					if (fwdWarpedIDDistFinal[idx - 1 + w1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1 + w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x - 1, y + 1);
					}
					if (fwdWarpedIDDistFinal[idx - 1 - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1 - w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x - 1, y - 1);
					}
					if (fwdWarpedIDDistFinal[idx + 1 - w1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1 - w1] = k;
						bfsList1[bfsNum++] = Eigen::Vector2i(x + 1, y - 1);
					}
				}
			}
		}
	}

	void CoarseDistanceMap::addIntoDistFinal(int u, int v)
	{
		if (w[0] == 0) return;
		bfsList1[0] = Eigen::Vector2i(u, v);
		fwdWarpedIDDistFinal[u + w[1] * v] = 0;
		growDistBFS(1);
	}

	void CoarseDistanceMap::makeK(CalibHessian* HCalib)
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
}