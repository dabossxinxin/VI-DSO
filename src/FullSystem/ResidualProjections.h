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

#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"

#include "util/settings.h"
#include "util/NumType.h"

namespace dso
{
	/// <summary>
	/// 光度观测残差对逆深度求一阶导数
	/// </summary>
	/// <param name="t">输入：hostToTarget平移分量</param>
	/// <param name="u">输入：target帧中归一化相机坐标</param>
	/// <param name="v">输入：target帧中归一化相机坐标</param>
	/// <param name="dx">输入：特征点像素坐标扰动，这里没有用到</param>
	/// <param name="dy">输入：特征点像素坐标扰动，这里没有用到</param>
	/// <param name="dxInterp">输入：target帧中x方向梯度与相机内参fx的乘积</param>
	/// <param name="dyInterp">输入：target帧中y方向梯度与相机内参fy的乘积</param>
	/// <param name="drescale">输入：id_host^-1*id_target</param>
	/// <returns>光度残差对逆深度的一阶导数</returns>
	EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
	{
		return (dxInterp*drescale * (t[0] - t[2] * u)
			+ dyInterp * drescale * (t[1] - t[2] * v))*SCALE_IDEPTH;
	}

	/// <summary>
	/// 将host帧中的像素坐标投影到target帧中
	/// </summary>
	/// <param name="u_pt">输入：host帧中像素坐标</param>
	/// <param name="v_pt">输入：host帧中像素坐标</param>
	/// <param name="idepth">输入：host帧中逆深度</param>
	/// <param name="KRKi">输入：hostToTarget旋转分量</param>
	/// <param name="Kt">输入：hostToTarget平移分量</param>
	/// <param name="Ku">输出：target帧中像素坐标</param>
	/// <param name="Kv">输出：target帧中像素坐标</param>
	/// <returns>特征投影到target帧后是否越界</returns>
	EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv)
	{
		Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
		Ku = ptp[0] / ptp[2];
		Kv = ptp[1] / ptp[2];
		return Ku > 1.1f && Kv > 1.1f && Ku < wM3G&& Kv < hM3G;
	}

	/// <summary>
	/// 将host帧中的像素坐标投影到target帧像素坐标中
	/// </summary>
	/// <param name="u_pt">输入：host帧中像素坐标</param>
	/// <param name="v_pt">输入：host帧中像素坐标</param>
	/// <param name="idepth">输入：特征在host帧中的逆深度</param>
	/// <param name="dx">输入：特征像素点u方向扰动</param>
	/// <param name="dy">输入：特征像素点v方向扰动</param>
	/// <param name="HCalib">输入：相机内参信息</param>
	/// <param name="R">输入：hostToTarget旋转矩阵</param>
	/// <param name="t">输入：hostToTarget平移矩阵</param>
	/// <param name="drescale">输出：id_host^-1*id_target</param>
	/// <param name="u">输出：target帧中归一化相机坐标</param>
	/// <param name="v">输出：target帧中归一化相机坐标</param>
	/// <param name="Ku">输出：target帧中像素坐标</param>
	/// <param name="Kv">输出：target帧中像素坐标</param>
	/// <param name="KliP">输出：host帧中归一化相机坐标</param>
	/// <param name="new_idepth">输出：target帧中逆深度</param>
	/// <returns>特征点在target帧中投影是否越界</returns>
	EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
	{
		// host帧中归一化相机系坐标
		KliP = Vec3f(
			(u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
			(v_pt + dy - HCalib->cyl()) * HCalib->fyli(),
			1);

		Vec3f ptp = R * KliP + t * idepth;
		drescale = 1.0f / ptp[2];
		new_idepth = idepth * drescale;

		if (!(drescale > 0)) return false;
		
		// target帧中归一化相机系坐标
		u = ptp[0] * drescale;
		v = ptp[1] * drescale;

		// target帧中像素坐标
		Ku = u * HCalib->fxl() + HCalib->cxl();
		Kv = v * HCalib->fyl() + HCalib->cyl();

		return Ku > 1.1f && Kv > 1.1f && Ku < wM3G&& Kv < hM3G;
	}
}