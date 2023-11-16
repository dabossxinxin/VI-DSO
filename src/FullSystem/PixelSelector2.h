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
 
#include "util/NumType.h"
#include "FullSystem/HessianBlocks.h"

namespace dso
{
	enum PixelSelectorStatus
	{
		PIXSEL_VOID = 0,
		PIXSEL_1 = 1,
		PIXSEL_2 = 2,
		PIXSEL_3 = 3
	};

	/// <summary>
	/// 在图像中提取特征点：为了保证所选特征的均匀性，执行的是分块选点的策略
	/// 选出来的像素点记录在statusMap中，值为1表示在第一层金字塔中选的点，2为第二层，4为第三层
	/// </summary>
	class PixelSelector
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		PixelSelector(int w, int h);
		~PixelSelector();

		int makeMaps(const FrameHessian* const fh, float* map_out, float density,
			int recursionsLeft = 1, bool plot = false, float thFactor = 1);

		void makeHists(const FrameHessian* const fh);

		int currentPotential;
		bool allowFast;
	private:

		Eigen::Vector3i select(const FrameHessian* const fh,
			float* map_out, int pot, float thFactor = 1);

		unsigned char* randomPattern;
		int* gradHist;
		float* ths;
		float* thsSmoothed;
		int thsStep;
		const FrameHessian* gradHistFrame;
	};
}