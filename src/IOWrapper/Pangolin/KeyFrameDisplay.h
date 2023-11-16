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
		/// ����״̬/���
		/// </summary>
		enum InputPointStatus
		{
			STATUS_IMMATURE = 0,
			STATUS_NORMAL = 1,
			STATUS_MARGINALIZE = 2,
			STATUS_OUTLIER = 3
		};
		
		/// <summary>
		/// ������ʾ�̵߳�������ṹ
		/// </summary>
		template<int ppp>
		struct InputPointSparse
		{
			float u;					// ��������������
			float v;					// ��������������
			float idpeth;				// ����������֡�������
			float idepth_hessian;		// ���������й۲���Ϣ��hessian
			float relObsBaseline;		// ���������й۲��е������߳���
			int numGoodRes;				// ���������й۲��кõĹ۲������
			unsigned char color[ppp];	// ����������֡ͼ���е����ػҶ�ֵ
			InputPointStatus status;	// ������״̬/���
		};
		typedef InputPointSparse<MAX_RES_PER_POINT> InputPointSparseGroup;

		/// <summary>
		/// ��������ʾ�߳��л��ƹؼ�֡λ���Լ������������
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

			int id;										// ��ǰ֡ID
			SE3 camToWorld;								// ��ǰ֡λ��

			inline bool operator < (const KeyFrameDisplay& other) const
			{
				return (id < other.id);
			}

		private:
			float fx, fy, cx, cy;						// ����ڲ���Ϣ
			float fxi, fyi, cxi, cyi;					// ����ڲ���Ϣ
			int width, height;							// ���ͼ���С��Ϣ

			float my_scaledTH, my_absTH, my_scale;		// ��ǰ�趨�Ĺؼ�֡����ɸѡ����
			int my_sparsifyFactor;						// ͨ��ϡ��̶�ɸѡ�ؼ�֡�е�����������ʾ
			int my_displayMode;							// ͨ����ʾģʽɸѡ�ؼ�֡�е�����������ʾ
			float my_minRelBS;							// ͨ����������ɸѡ�ؼ�֡�е�����������ʾ
			bool needRefresh;							// �ؼ�֡�е���ʾ�����Ƿ���Ҫˢ��

			int numSparsePoints;						// originalInputSparseʵ�ʴ洢���ݿռ��С
			int numSparseBufferSize;					// originalInputSparse���������ڴ�ռ��С
			InputPointSparseGroup* originalInputSparse;	// �ؼ�֡�й���������������ݣ�����δ����㡢�����㡢��Ե�����Լ����

			bool bufferValid;							// ����ʾ��buffer�����Ƿ�׼����
			int numGLBufferPoints;						// ����ʾ��buffer���������ڴ�ռ��С
			int numGLBufferGoodPoints;					// ����ʾ��bufferʵ�ʴ洢���ݿռ��С
			pangolin::GlBuffer vertexBuffer;			// ���ڴ洢����ʾ��������������
			pangolin::GlBuffer colorBuffer;				// ���ڴ洢����ʾ��������ɫ����
		};
	}
}