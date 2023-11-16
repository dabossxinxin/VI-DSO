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

#undef Success

#include <sstream>
#include <fstream>

#include <Eigen/Core>

#include <pangolin/pangolin.h>

#include "util/NumType.h"

namespace dso
{
	class CalibHessian;
	class FrameHessian;
	class FrameShell;

	namespace IOWrap
	{
		/// <summary>
		/// 特征状态/类别
		/// </summary>
		enum InputPointStatus
		{
			STATUS_IMMATURE = 0,
			STATUS_NORMAL = 1,
			STATUS_MARGINALIZE = 2,
			STATUS_OUTLIER = 3
		};
		
		/// <summary>
		/// 输入显示线程的特征点结构
		/// </summary>
		template<int ppp>
		struct InputPointSparse
		{
			float u;					// 特征点像素坐标
			float v;					// 特征点像素坐标
			float idpeth;				// 特征点在主帧的逆深度
			float idepth_hessian;		// 特征点所有观测信息的hessian
			float relObsBaseline;		// 特征点所有观测中的最大基线长度
			int numGoodRes;				// 特征点所有观测中好的观测的数量
			unsigned char color[ppp];	// 特征点在主帧图像中的像素灰度值
			InputPointStatus status;	// 特征点状态/类别
		};
		typedef InputPointSparse<MAX_RES_PER_POINT> InputPointSparseGroup;

		/// <summary>
		/// 用于在显示线程中绘制关键帧位姿以及管理的特征点
		/// </summary>
		class KeyFrameDisplay
		{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
			KeyFrameDisplay();
			~KeyFrameDisplay();

			// copies points from KF over to internal buffer,
			// keeping some additional information so we can render it differently.
			void setFromKF(FrameHessian* fh, CalibHessian* HCalib);

			// copies points from KF over to internal buffer,
			// keeping some additional information so we can render it differently.
			void setFromF(FrameShell* fs, CalibHessian* HCalib);

			// copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
			bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

			// renders cam & pointcloud.
			void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor = 1);
			void drawPC(float pointSize);

			int validDisplayPts() { return numGLBufferGoodPoints; }

			int id;										// 当前帧ID
			SE3 camToWorld;								// 当前帧位姿

			inline bool operator < (const KeyFrameDisplay& other) const
			{
				return (id < other.id);
			}

		private:
			float fx, fy, cx, cy;						// 相机内参信息
			float fxi, fyi, cxi, cyi;					// 相机内参信息
			int width, height;							// 相机图像大小信息

			float my_scaledTH, my_absTH, my_scale;		// 当前设定的关键帧特征筛选参数
			int my_sparsifyFactor;						// 通过稀疏程度筛选关键帧中的特征用于显示
			int my_displayMode;							// 通过显示模式筛选关键帧中的特征用于显示
			float my_minRelBS;							// 通过特征基线筛选关键帧中的特征用于显示
			bool needRefresh;							// 关键帧中的显示数据是否需要刷新

			int numSparsePoints;						// originalInputSparse实际存储数据空间大小
			int numSparseBufferSize;					// originalInputSparse数据声明内存空间大小
			InputPointSparseGroup* originalInputSparse;	// 关键帧中管理的所有特征数据：包含未成熟点、正常点、边缘化点以及外点

			bool bufferValid;							// 待显示的buffer数据是否准备好
			int numGLBufferPoints;						// 待显示的buffer数据声明内存空间大小
			int numGLBufferGoodPoints;					// 待显示的buffer实际存储数据空间大小
			pangolin::GlBuffer vertexBuffer;			// 用于存储待显示特征的坐标数据
			pangolin::GlBuffer colorBuffer;				// 用于存储待显示特征的颜色数据
		};
	}
}