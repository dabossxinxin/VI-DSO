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
#include <stdio.h>
#include <pangolin/pangolin.h>

#include "KeyFrameDisplay.h"

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"

#include "util/settings.h"
#include "util/FrameShell.h"

namespace dso
{
	namespace IOWrap
	{
		/// <summary>
		/// KeyFrameDisplay构造函数
		/// </summary>
		KeyFrameDisplay::KeyFrameDisplay()
		{
			originalInputSparse = 0;
			numSparseBufferSize = 0;
			numSparsePoints = 0;

			id = 0;
			//active = true;
			camToWorld = SE3();

			needRefresh = true;

			my_scaledTH = 1e10;
			my_absTH = 1e10;
			my_displayMode = 1;
			my_minRelBS = 0;
			my_sparsifyFactor = 1;

			numGLBufferPoints = 0;
			numGLBufferGoodPoints = 0;
			bufferValid = false;

			fx = fxi = 0;
			fy = fyi = 0;
			cx = cxi = 0;
			cy = cyi = 0;
			width = height = 0;
		}

		/// <summary>
		/// 向KeyFrameDisplay对象中输入帧位姿数据以及相机内参数据
		/// </summary>
		/// <param name="frame">帧位姿数据</param>
		/// <param name="HCalib">相机内参数据</param>
		void KeyFrameDisplay::setFromF(FrameShell* frame, CalibHessian* HCalib)
		{
			id = frame->id;
			fx = HCalib->fxl();
			fy = HCalib->fyl();
			cx = HCalib->cxl();
			cy = HCalib->cyl();
			width = wG[0];
			height = hG[0];
			fxi = 1 / fx;
			fyi = 1 / fy;
			cxi = -cx / fx;
			cyi = -cy / fy;
			camToWorld = frame->camToWorld;
			needRefresh = true;
		}

		/// <summary>
		/// 向KeyFrameDisplay设置关键帧数据以及相机内参数据
		/// </summary>
		/// <param name="fh">关键帧数据</param>
		/// <param name="HCalib">相机内参数据</param>
		void KeyFrameDisplay::setFromKF(FrameHessian* fh, CalibHessian* HCalib)
		{
			setFromF(fh->shell, HCalib);

			int npoints = fh->immaturePoints.size() +
				fh->pointHessians.size() +
				fh->pointHessiansMarginalized.size() +
				fh->pointHessiansOut.size();

			if (numSparseBufferSize < npoints)
			{
				SAFE_DELETE(originalInputSparse, true);
				numSparseBufferSize = npoints + 100;
				originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
			}

			auto pc = originalInputSparse;
			numSparsePoints = 0;
			for (ImmaturePoint* p : fh->immaturePoints)
			{
				for (int it = 0; it < patternNum; ++it)
					pc[numSparsePoints].color[it] = p->color[it];

				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = (p->idepth_max + p->idepth_min) * 0.5f;
				pc[numSparsePoints].idepth_hessian = 1000;
				pc[numSparsePoints].relObsBaseline = 0;
				pc[numSparsePoints].numGoodRes = 1;
				pc[numSparsePoints].status = InputPointStatus::STATUS_IMMATURE;
				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessians)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::STATUS_NORMAL;

				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessiansMarginalized)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::STATUS_MARGINALIZE;
				numSparsePoints++;
			}

			for (PointHessian* p : fh->pointHessiansOut)
			{
				for (int i = 0; i < patternNum; i++)
					pc[numSparsePoints].color[i] = p->color[i];
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = InputPointStatus::STATUS_OUTLIER;
				numSparsePoints++;
			}
			assert(numSparsePoints <= npoints);

			camToWorld = fh->PRE_camToWorld;
			needRefresh = true;
		}

		/// <summary>
		/// 析构KeyFrameDisplay中成员变量内存空间
		/// </summary>
		KeyFrameDisplay::~KeyFrameDisplay()
		{
			SAFE_DELETE(originalInputSparse,true);
		}

		/// <summary>
		/// 刷新用于存储显示数据的内存空间，包含vertexBuffer和colorBuffer
		/// </summary>
		/// <param name="canRefresh">外部给定的是否可以刷新的标志位</param>
		/// <param name="scaledTH">可能是逆深度变异系数的阈值</param>
		/// <param name="absTH">逆深度数据方差阈值</param>
		/// <param name="mode">特征显示模式</param>
		/// <param name="minBS">需显示特征的最小基线长度</param>
		/// <param name="sparsity">需显示特征的稀疏程度</param>
		/// <returns>是否刷新成功</returns>
		bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
		{
			if (canRefresh)
			{
				needRefresh = needRefresh ||
					my_scaledTH != scaledTH ||
					my_absTH != absTH ||
					my_displayMode != mode ||
					my_minRelBS != minBS ||
					my_sparsifyFactor != sparsity;
			}

			if (!needRefresh) return false;
			needRefresh = false;

			my_scaledTH = scaledTH;
			my_absTH = absTH;
			my_displayMode = mode;
			my_minRelBS = minBS;
			my_sparsifyFactor = sparsity;

			// if there are no vertices, done!
			if (numSparsePoints == 0)
				return false;

			// 构造记录特征点坐标数据以及颜色数据的指针
			Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints * patternNum];
			Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints * patternNum];
			int vertexBufferNumPoints = 0;

			for (int it = 0; it < numSparsePoints; ++it)
			{
				/* display modes:
				 * my_displayMode==0 - 显示关键帧中所有特征
				 * my_displayMode==1 - 显示关键帧中正常特征：激活特征和边缘化特征
				 * my_displayMode==2 - 显示关键帧中激活特征
				 * my_displayMode==3 - 不显示关键帧特征
				 */
				if (my_displayMode == 1 && originalInputSparse[it].status != InputPointStatus::STATUS_NORMAL
					&& originalInputSparse[it].status != InputPointStatus::STATUS_MARGINALIZE) continue;
				if (my_displayMode == 2 && originalInputSparse[it].status != InputPointStatus::STATUS_NORMAL) continue;
				if (my_displayMode > 2) continue;

				if (originalInputSparse[it].idpeth < 0) continue;

				float depth = 1.0f / originalInputSparse[it].idpeth;
				float depth4 = std::pow(depth, 4);
				float var = (1.0f / (originalInputSparse[it].idepth_hessian + 0.01));

				// TODO:像是变异系数，但是又多除以了depth2
				if (var * depth4 > my_scaledTH)
					continue;

				// 特征逆深度方差越小，该特征逆深度精度越高
				if (var > my_absTH)
					continue;

				// 特征基线长度越长，匹配过程中误匹配带来的误差越小
				if (originalInputSparse[it].relObsBaseline < my_minRelBS)
					continue;

				for (int pnt = 0; pnt < patternNum; pnt++)
				{
					if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0) continue;
					int dx = patternP[pnt][0];
					int dy = patternP[pnt][1];

					tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[it].u + dx) * fxi + cxi) * depth;
					tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[it].v + dy) * fyi + cyi) * depth;
					tmpVertexBuffer[vertexBufferNumPoints][2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));

					// 若显示模式为显示关键帧中所有特征，则设定不同的颜色显示关键帧中不同类型特征
					// 若显示模式为显示关键帧中已经激活且并未Outlier的特征，将特征颜色设置为像素灰度值
					if (my_displayMode == 0)
					{
						if (originalInputSparse[it].status == InputPointStatus::STATUS_IMMATURE)
						{
							tmpColorBuffer[vertexBufferNumPoints][0] = 0;
							tmpColorBuffer[vertexBufferNumPoints][1] = 255;
							tmpColorBuffer[vertexBufferNumPoints][2] = 255;
						}
						else if (originalInputSparse[it].status == InputPointStatus::STATUS_NORMAL)
						{
							tmpColorBuffer[vertexBufferNumPoints][0] = 0;
							tmpColorBuffer[vertexBufferNumPoints][1] = 255;
							tmpColorBuffer[vertexBufferNumPoints][2] = 0;
						}
						else if (originalInputSparse[it].status == InputPointStatus::STATUS_MARGINALIZE)
						{
							tmpColorBuffer[vertexBufferNumPoints][0] = 0;
							tmpColorBuffer[vertexBufferNumPoints][1] = 0;
							tmpColorBuffer[vertexBufferNumPoints][2] = 255;
						}
						else if (originalInputSparse[it].status == InputPointStatus::STATUS_OUTLIER)
						{
							tmpColorBuffer[vertexBufferNumPoints][0] = 255;
							tmpColorBuffer[vertexBufferNumPoints][1] = 0;
							tmpColorBuffer[vertexBufferNumPoints][2] = 0;
						}
						else
						{
							tmpColorBuffer[vertexBufferNumPoints][0] = 255;
							tmpColorBuffer[vertexBufferNumPoints][1] = 255;
							tmpColorBuffer[vertexBufferNumPoints][2] = 255;
						}
					}
					else
					{
						tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[it].color[pnt];
						tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[it].color[pnt];
						tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[it].color[pnt];
					}
					vertexBufferNumPoints++;

					assert(vertexBufferNumPoints <= numSparsePoints * patternNum);
				}
			}

			if (vertexBufferNumPoints == 0)
			{
				SAFE_DELETE(tmpColorBuffer,true);
				SAFE_DELETE(tmpVertexBuffer,true);
				return true;
			}

			// 刷新后，如果刷新前申明的内存空间以及足够放置特征数据，那么就不需要重新开辟内存空间
			numGLBufferGoodPoints = vertexBufferNumPoints;
			if (numGLBufferGoodPoints > numGLBufferPoints)
			{
				numGLBufferPoints = vertexBufferNumPoints * 1.3;
				vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
				colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
			}
			vertexBuffer.Upload(tmpVertexBuffer, sizeof(float) * 3 * numGLBufferGoodPoints, 0);
			colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char) * 3 * numGLBufferGoodPoints, 0);
			bufferValid = true;

			SAFE_DELETE(tmpColorBuffer, true);
			SAFE_DELETE(tmpVertexBuffer, true);

			return true;
		}

		/// <summary>
		/// 绘制关键帧的位姿
		/// </summary>
		/// <param name="lineWidth">绘制对象线宽</param>
		/// <param name="color">绘制对象颜色</param>
		/// <param name="sizeFactor">绘制对象大小比例</param>
		void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
		{
			if (width == 0)
				return;

			float sz = sizeFactor;

			glPushMatrix();

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			glMultMatrixf((GLfloat*)m.data());

			if (color == 0) glColor3f(1, 0, 0);
			else glColor3f(color[0], color[1], color[2]);

			glLineWidth(lineWidth);
			glBegin(GL_LINES);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

			glEnd();
			glPopMatrix();
		}

		/// <summary>
		/// 显示colorBuffer和vertexBuffer中显示的特征信息
		/// </summary>
		/// <param name="pointSize">特征点渲染尺寸</param>
		void KeyFrameDisplay::drawPC(float pointSize)
		{
			if (!bufferValid || numGLBufferGoodPoints == 0)
				return;

			glDisable(GL_LIGHTING);

			glPushMatrix();

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			glMultMatrixf((GLfloat*)m.data());

			glPointSize(pointSize);

			colorBuffer.Bind();
			glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			vertexBuffer.Bind();
			glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
			glDisableClientState(GL_VERTEX_ARRAY);
			vertexBuffer.Unbind();

			glDisableClientState(GL_COLOR_ARRAY);
			colorBuffer.Unbind();

			glPopMatrix();
		}
	}
}