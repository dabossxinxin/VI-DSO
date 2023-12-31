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
#pragma once

#include <vector>
#include <math.h>
 
#include "util/NumType.h"
#include "util/IndexThreadReduce.h"

#include "OptimizationBackend/MatrixAccumulators.h"

namespace dso
{
	class EFPoint;
	class EnergyFunctional;

	/// <summary>
	/// 计算Hessian矩阵的舒尔补
	/// </summary>
	class AccumulatedSCHessianSSE
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		inline AccumulatedSCHessianSSE()
		{
			for (int it = 0; it < NUM_THREADS; ++it)
			{
				accE[it] = 0;
				accEB[it] = 0;
				accD[it] = 0;
				nframes[it] = 0;
			}
		};

		inline ~AccumulatedSCHessianSSE()
		{
			for (int it = 0; it < NUM_THREADS; ++it)
			{
				if (accE[it] != 0) delete[] accE[it];
				if (accEB[it] != 0) delete[] accEB[it];
				if (accD[it] != 0) delete[] accD[it];
			}
		};

		inline void setZero(int n, int min = 0, int max = 1, Vec10* stats = 0, int tid = 0)
		{
			// 为成员变量开辟内存空间
			if (n != nframes[tid])
			{
				if (accE[tid] != 0) delete[] accE[tid];
				if (accEB[tid] != 0) delete[] accEB[tid];
				if (accD[tid] != 0) delete[] accD[tid];
				accE[tid] = new AccumulatorXX<8, CPARS>[n * n];
				accEB[tid] = new AccumulatorX<8>[n * n];
				accD[tid] = new AccumulatorXX<8, 8>[n * n * n];
			}
			accbc[tid].initialize();
			accHcc[tid].initialize();

			for (int i = 0; i < n * n; i++)
			{
				accE[tid][i].initialize();
				accEB[tid][i].initialize();

				for (int j = 0; j < n; j++)
					accD[tid][i * n + j].initialize();
			}
			nframes[tid] = n;
		}

		void stitchDouble(MatXX& H_sc, VecX& b_sc, EnergyFunctional const* const EF, int tid = 0);
		void addPoint(EFPoint* p, bool shiftPriorToZero, int tid = 0);

		void stitchDoubleMT(IndexThreadReduce<Vec10>* red, MatXX& H, VecX& b, EnergyFunctional const* const EF, bool MT)
		{
			if (MT)
			{
				MatXX Hs[NUM_THREADS];
				VecX bs[NUM_THREADS];
				for (int it = 0; it < NUM_THREADS; ++it)
				{
					assert(nframes[0] == nframes[it]);
					Hs[it] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
					bs[it] = VecX::Zero(nframes[0] * 8 + CPARS);
				}

				red->reduce(std::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal, this, Hs, bs, EF,
					std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, nframes[0] * nframes[0], 0);

				H = Hs[0];
				b = bs[0];

				for (int it = 1; it < NUM_THREADS; ++it)
				{
					H.noalias() += Hs[it];
					b.noalias() += bs[it];
				}
			}
			else
			{
				H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
				b = VecX::Zero(nframes[0] * 8 + CPARS);
				stitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
			}

			for (int h = 0; h < nframes[0]; ++h)
			{
				int hIdx = CPARS + h * 8;
				H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
			}
		}

		AccumulatorXX<8, CPARS>*	accE[NUM_THREADS];			// 令p表示帧位姿和光度参数，项Hpd*Hdd^-1*Hdc计算handle
		AccumulatorX<8>*			accEB[NUM_THREADS];			// 令p表示帧位姿和光度参数，项Hpd*Hdd^-1*bd计算handle
		AccumulatorXX<8, 8>*		accD[NUM_THREADS];			// 令p表示帧位姿和光度参数，项Hpd*Hdd^-1*Hdp计算handle
		AccumulatorXX<CPARS, CPARS> accHcc[NUM_THREADS];		// 项Hcd*Hdd^-1*Hdc计算handle
		AccumulatorX<CPARS>			accbc[NUM_THREADS];			// 项Hcd*Hdd^-1*bd计算handle
		int							nframes[NUM_THREADS];		// 每个线程计算中滑窗关键帧的数量

		void addPointsInternal(std::vector<EFPoint*>* points, bool shiftPriorToZero, int min = 0, int max = 1, Vec10* stats = 0, int tid = 0)
		{
			for (int it = min; it < max; ++it)
				addPoint((*points)[it], shiftPriorToZero, tid);
		}

	private:

		void stitchDoubleInternal(MatXX* H, VecX* b, EnergyFunctional const* const EF, int min, int max, Vec10* stats, int tid);
	};
}