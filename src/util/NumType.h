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

#include "Eigen/Core"
#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace dso
{
// CAMERA MODEL TO USE
#define SSEE(val,idx) (*(((float*)&val)+idx))

#define MAX_RES_PER_POINT 8
#define NUM_THREADS 6

#define todouble(x) (x).cast<double>()

#define CPARS 4

#define MatToDynamic(x) MatXX(x)

	typedef Sophus::SE3d SE3;
	typedef Sophus::Sim3d Sim3;
	typedef Sophus::SO3d SO3;
	typedef Sophus::SO3f SO3f;
	typedef Sophus::RxSO3d RxSO3;

	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
	typedef Eigen::Matrix<double, CPARS, CPARS> MatCC;

	typedef Eigen::Matrix<double, CPARS, 10> MatC10;
	typedef Eigen::Matrix<double, 10, 10> Mat1010;
	typedef Eigen::Matrix<double, 13, 13> Mat1313;

	typedef Eigen::Matrix<double, 8, 10> Mat810;
	typedef Eigen::Matrix<double, 8, 3> Mat83;
	typedef Eigen::Matrix<double, 6, 6> Mat66;
	typedef Eigen::Matrix<double, 9, 6> Mat96;
	typedef Eigen::Matrix<double, 5, 3> Mat53;
	typedef Eigen::Matrix<double, 4, 3> Mat43;
	typedef Eigen::Matrix<double, 4, 2> Mat42;
	typedef Eigen::Matrix<double, 3, 3> Mat33;
	typedef Eigen::Matrix<double, 3, 6> Mat36;
	typedef Eigen::Matrix<double, 2, 2> Mat22;
	typedef Eigen::Matrix<double, 8, CPARS> Mat8C;
	typedef Eigen::Matrix<double, CPARS, 8> MatC8;
	typedef Eigen::Matrix<float, 8, CPARS> Mat8Cf;
	typedef Eigen::Matrix<float, CPARS, 8> MatC8f;

	typedef Eigen::Matrix<double, 8, 8> Mat88;
	typedef Eigen::Matrix<double, 7, 7> Mat77;
	typedef Eigen::Matrix<double, 7, 3> Mat73;

	typedef Eigen::Matrix<double, CPARS, 1> VecC;
	typedef Eigen::Matrix<float, CPARS, 1> VecCf;
	typedef Eigen::Matrix<double, 15, 1> Vec15;
	typedef Eigen::Matrix<double, 17, 1> Vec17;
	typedef Eigen::Matrix<double, 16, 1> Vec16;
	typedef Eigen::Matrix<double, 13, 1> Vec13;
	typedef Eigen::Matrix<double, 12, 1> Vec12;
	typedef Eigen::Matrix<double, 11, 1> Vec11;
	typedef Eigen::Matrix<double, 10, 1> Vec10;
	typedef Eigen::Matrix<double, 9, 1> Vec9;
	typedef Eigen::Matrix<double, 8, 1> Vec8;
	typedef Eigen::Matrix<double, 7, 1> Vec7;
	typedef Eigen::Matrix<double, 6, 1> Vec6;
	typedef Eigen::Matrix<double, 5, 1> Vec5;
	typedef Eigen::Matrix<double, 4, 1> Vec4;
	typedef Eigen::Matrix<double, 3, 1> Vec3;
	typedef Eigen::Matrix<double, 2, 1> Vec2;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

	typedef Eigen::Matrix<float, 3, 3> Mat33f;
	typedef Eigen::Matrix<float, 10, 3> Mat103f;
	typedef Eigen::Matrix<float, 2, 2> Mat22f;
	typedef Eigen::Matrix<float, 3, 1> Vec3f;
	typedef Eigen::Matrix<float, 2, 1> Vec2f;
	typedef Eigen::Matrix<float, 6, 1> Vec6f;

	typedef Eigen::Matrix<double, 4, 9> Mat49;
	typedef Eigen::Matrix<double, 8, 9> Mat89;

	typedef Eigen::Matrix<double, 9, 4> Mat94;
	typedef Eigen::Matrix<double, 9, 8> Mat98;

	typedef Eigen::Matrix<double, 8, 1> Mat81;
	typedef Eigen::Matrix<double, 1, 8> Mat18;
	typedef Eigen::Matrix<double, 9, 1> Mat91;
	typedef Eigen::Matrix<double, 1, 9> Mat19;

	typedef Eigen::Matrix<double, 8, 4> Mat84;
	typedef Eigen::Matrix<double, 4, 8> Mat48;
	typedef Eigen::Matrix<double, 4, 4> Mat44;

	typedef Eigen::Matrix<double, 9, 9> Mat99;
	typedef Eigen::Matrix<double, 9, 7> Mat97;
	typedef Eigen::Matrix<double, 9, 3> Mat93;
	typedef Eigen::Matrix<double, 12, 12> Mat1212;
	typedef Eigen::Matrix<double, 11, 11> Mat1111;
	typedef Eigen::Matrix<double, 15, 15> Mat1515;
	typedef Eigen::Matrix<double, 17, 17> Mat1717;
	typedef Eigen::Matrix<double, 16, 16> Mat1616;
	typedef Eigen::Matrix<double, 10, 15> Mat1015;
	typedef Eigen::Matrix<double, 9, 15> Mat915;
	typedef Eigen::Matrix<double, 9, 16> Mat916;

	typedef Eigen::Matrix<float, MAX_RES_PER_POINT, 1> VecNRf;
	typedef Eigen::Matrix<float, 12, 1> Vec12f;
	typedef Eigen::Matrix<float, 1, 8> Mat18f;
	typedef Eigen::Matrix<float, 6, 6> Mat66f;
	typedef Eigen::Matrix<float, 8, 8> Mat88f;
	typedef Eigen::Matrix<float, 8, 4> Mat84f;
	typedef Eigen::Matrix<float, 8, 1> Vec8f;
	typedef Eigen::Matrix<float, 10, 1> Vec10f;
	typedef Eigen::Matrix<float, 6, 6> Mat66f;
	typedef Eigen::Matrix<float, 4, 1> Vec4f;
	typedef Eigen::Matrix<float, 4, 4> Mat44f;
	typedef Eigen::Matrix<float, 12, 12> Mat1212f;
	typedef Eigen::Matrix<float, 12, 1> Vec12f;
	typedef Eigen::Matrix<float, 13, 13> Mat1313f;
	typedef Eigen::Matrix<float, 10, 10> Mat1010f;
	typedef Eigen::Matrix<float, 13, 1> Vec13f;
	typedef Eigen::Matrix<float, 9, 9> Mat99f;
	typedef Eigen::Matrix<float, 9, 1> Vec9f;

	typedef Eigen::Matrix<float, 4, 2> Mat42f;
	typedef Eigen::Matrix<float, 6, 2> Mat62f;
	typedef Eigen::Matrix<float, 1, 2> Mat12f;
	typedef Eigen::Matrix<float, 3, 2> Mat32f;
	typedef Eigen::Matrix<float, 3, 6> Mat36f;

	typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;

	typedef Eigen::Matrix<double, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPC;
	typedef Eigen::Matrix<float, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPCf;
	typedef Eigen::Matrix<double, 8 + CPARS + 1, 1> VecPC;
	typedef Eigen::Matrix<float, 8 + CPARS + 1, 1> VecPCf;

	typedef Eigen::Matrix<float, 14, 14> Mat1414f;
	typedef Eigen::Matrix<float, 14, 1> Vec14f;
	typedef Eigen::Matrix<double, 14, 14> Mat1414;
	typedef Eigen::Matrix<double, 14, 1> Vec14;

	/// <summary>
	/// 光度参数结构
	/// </summary>
	struct AffLight
	{
		AffLight(double a_, double b_) : a(a_), b(b_) {};
		AffLight() : a(0), b(0) {};

		// Affine Parameters:
		// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).
		double a, b;	

		static Vec2 fromToVecExposure(float exposureH, float exposureT, AffLight g2H, AffLight g2T)
		{
			if (exposureH == 0 || exposureT == 0)
			{
				exposureT = exposureH = 1;
			}

			double a = exp(g2T.a - g2H.a) * exposureT / exposureH;
			double b = g2T.b - a * g2H.b;
			return Vec2(a, b);
		}

		Vec2 vec()
		{
			return Vec2(a, b);
		}
	};

	/// <summary>
	/// 残差状态枚举值
	/// </summary>
	enum ResState
	{
		INNER = 0,		// 残差的值小于阈值视为内点
		OOB,			// 点不在主帧或目标帧视野中
		OUTLIER			// 残差的值大于阈值视为外点
	};

	/// <summary>
	/// 析构指针内存空间
	/// </summary>
	/// <typeparam name="T">指针类型</typeparam>
	/// <param name="data">指针地址</param>
	/// <param name="flag">是否为指针序列</param>
	template <typename T>
	void SAFE_DELETE(T* data, bool flag = false)
	{
		if (flag && data != NULL)
		{
			delete[] data;
			data = NULL;
		}
		else if (!flag && data != NULL)
		{
			delete data;
			data = NULL;
		}
	}
}