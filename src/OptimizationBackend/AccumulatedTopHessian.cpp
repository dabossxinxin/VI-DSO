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
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
	// 0 = active, 1 = linearized, 2=marginalize
	template<int mode>
	void AccumulatedTopHessianSSE::addPoint(EFPoint* p, EnergyFunctional const* const ef, int tid)
	{
		assert(mode == 0 || mode == 1 || mode == 2);

		VecCf dc = ef->cDeltaF;
		float dd = p->deltaF;

		float bd_acc = 0;
		float Hdd_acc = 0;
		VecCf  Hcd_acc = VecCf::Zero();

		for (EFResidual* r : p->residualsAll)
		{
			if (mode == 0)
			{
				if (r->isLinearized || !r->isActive()) continue;
			}
			if (mode == 1)
			{
				if (!r->isLinearized || !r->isActive()) continue;
			}
			if (mode == 2)
			{
				if (!r->isActive()) continue;
				assert(r->isLinearized);
			}

			RawResidualJacobian* rJ = r->J;
			int htIDX = r->hostIDX + r->targetIDX * nframes[tid];	// target帧索引为行、host帧索引为列
			Mat18f dp = ef->adHTdeltaF[htIDX];

			VecNRf resApprox;
			if (mode == 0)
				resApprox = rJ->resF;
			if (mode == 2)
				resApprox = r->res_toZeroF;
			if (mode == 1)
			{
				// compute Jp*delta
				__m128 Jp_delta_x = _mm_set1_ps(rJ->Jpdxi[0].dot(dp.head<6>()) +
					rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);				// 特征target帧u轴像素坐标对参数的雅可比与参数增量的乘积
				__m128 Jp_delta_y = _mm_set1_ps(rJ->Jpdxi[1].dot(dp.head<6>()) +
					rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);				// 特征target帧v轴像素坐标对参数的雅可比与参数增量的乘积		
				__m128 delta_a = _mm_set1_ps((float)(dp[6]));				// 光度变换参数a21增量
				__m128 delta_b = _mm_set1_ps((float)(dp[7]));				// 光度变换参数b21增量

				for (int it = 0; it < patternNum; it += 4)
				{
					// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
					__m128 rtz = _mm_load_ps(((float*)&r->res_toZeroF) + it);
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx)) + it), Jp_delta_x));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx + 1)) + it), Jp_delta_y));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF)) + it), delta_a));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF + 1)) + it), delta_b));
					_mm_store_ps(((float*)&resApprox) + it, rtz);
				}
			}

			// need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
			Vec2f JI_r(0, 0);
			Vec2f Jab_r(0, 0);
			float rr = 0;
			for (int it = 0; it < patternNum; ++it)
			{
				JI_r[0] += resApprox[it] * rJ->JIdx[0][it];
				JI_r[1] += resApprox[it] * rJ->JIdx[1][it];
				Jab_r[0] += resApprox[it] * rJ->JabF[0][it];
				Jab_r[1] += resApprox[it] * rJ->JabF[1][it];
				rr += resApprox[it] * resApprox[it];
			}

			// 更新Hessian矩阵左上角部分信息
			acc[tid][htIDX].update(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));

			// 更新Hessian矩阵右下角部分信息
			acc[tid][htIDX].updateBotRight(
				rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
				rJ->Jab2(1, 1), Jab_r[1], rr);

			// 更新Hessian矩阵右上角部分信息
			acc[tid][htIDX].updateTopRight(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
				rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1),
				JI_r[0], JI_r[1]);

			Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
			bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];
			Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
			Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];

			nres[tid]++;
		}

		// TODO：这里为什么要这样区分
		if (mode == 0)
		{
			p->Hdd_accAF = Hdd_acc;
			p->bd_accAF = bd_acc;
			p->Hcd_accAF = Hcd_acc;
		}
		if (mode == 1 || mode == 2)
		{
			p->Hdd_accLF = Hdd_acc;
			p->bd_accLF = bd_acc;
			p->Hcd_accLF = Hcd_acc;
		}
		if (mode == 2)
		{
			p->Hcd_accAF.setZero();
			p->Hdd_accAF = 0;
			p->bd_accAF = 0;
		}
	}

	template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint* p, EnergyFunctional const* const ef, int tid);
	template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const* const ef, int tid);
	template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint* p, EnergyFunctional const* const ef, int tid);

	void AccumulatedTopHessianSSE::stitchDouble(MatXX& H, VecX& b, EnergyFunctional const* const EF, bool usePrior, bool useDelta, int tid)
	{
		H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
		b = VecX::Zero(nframes[tid] * 8 + CPARS);

		for (int h = 0; h < nframes[tid]; ++h)
		{
			for (int t = 0; t < nframes[tid]; ++t)
			{
				int hIdx = CPARS + h * 8;
				int tIdx = CPARS + t * 8;
				int aidx = h + nframes[tid] * t;

				acc[tid][aidx].finish();
				if (acc[tid][aidx].num == 0) continue;

				MatPCPC accH = acc[tid][aidx].H.cast<double>();

				H.block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();
				H.block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
				H.block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

				H.block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);
				H.block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

				H.topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

				b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);
				b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);
				b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
			}
		}

		for (int h = 0; h < nframes[tid]; ++h)
		{
			int hIdx = CPARS + h * 8;
			H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

			for (int t = h + 1; t < nframes[tid]; ++t)
			{
				int tIdx = CPARS + t * 8;
				H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
				H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
			}
		}

		if (usePrior)
		{
			assert(useDelta);
			H.diagonal().head<CPARS>() += EF->cPrior;
			b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
			for (int h = 0; h < nframes[tid]; ++h)
			{
				H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
				b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
			}
		}
	}

	void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX* H, VecX* b, EnergyFunctional const* const EF, bool usePrior, int min, int max, Vec10* stats, int tid)
	{
		int toAggregate = NUM_THREADS;
		if (tid == -1)
		{
			toAggregate = 1;
			tid = 0;
		}
		if (min == max) return;

		for (int k = min; k < max; ++k)
		{
			int h = k % nframes[0];
			int t = k / nframes[0];

			int hIdx = CPARS + h * 8;
			int tIdx = CPARS + t * 8;
			int aidx = h + nframes[0] * t;

			assert(aidx == k);

			MatPCPC accH = MatPCPC::Zero();

			for (int tidx = 0; tidx < toAggregate; ++tidx)
			{
				acc[tidx][aidx].finish();
				if (acc[tidx][aidx].num == 0) continue;
				accH += acc[tidx][aidx].H.cast<double>();
			}

			// 将相机位姿以及相机光度相对量转化为绝对量，相机内参本身就是绝对量不用再使用伴随转换
			H[tid].block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();
			H[tid].block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
			H[tid].block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

			H[tid].block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);
			H[tid].block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

			H[tid].topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

			b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);
			b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

			b[tid].head<CPARS>().noalias() += accH.block<CPARS, 1>(0, CPARS + 8);
		}

		// 给Hessian矩阵添加先验，不需要每个线程都添加一遍，实际上先验信息只需要添加一遍就可以
		if (min == 0 && usePrior)
		{
			H[tid].diagonal().head<CPARS>() += EF->cPrior;
			b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
			for (int h = 0; h < nframes[tid]; ++h)
			{
				H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
				b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
			}
		}
	}
}