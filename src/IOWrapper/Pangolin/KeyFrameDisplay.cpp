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
		/// KeyFrameDisplay���캯��
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
		/// ��KeyFrameDisplay����������֡λ�������Լ�����ڲ�����
		/// </summary>
		/// <param name="frame">֡λ������</param>
		/// <param name="HCalib">����ڲ�����</param>
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
		/// ��KeyFrameDisplay���ùؼ�֡�����Լ�����ڲ�����
		/// </summary>
		/// <param name="fh">�ؼ�֡����</param>
		/// <param name="HCalib">����ڲ�����</param>
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
		/// ����KeyFrameDisplay�г�Ա�����ڴ�ռ�
		/// </summary>
		KeyFrameDisplay::~KeyFrameDisplay()
		{
			SAFE_DELETE(originalInputSparse,true);
		}

		/// <summary>
		/// ˢ�����ڴ洢��ʾ���ݵ��ڴ�ռ䣬����vertexBuffer��colorBuffer
		/// </summary>
		/// <param name="canRefresh">�ⲿ�������Ƿ����ˢ�µı�־λ</param>
		/// <param name="scaledTH">����������ȱ���ϵ������ֵ</param>
		/// <param name="absTH">��������ݷ�����ֵ</param>
		/// <param name="mode">������ʾģʽ</param>
		/// <param name="minBS">����ʾ��������С���߳���</param>
		/// <param name="sparsity">����ʾ������ϡ��̶�</param>
		/// <returns>�Ƿ�ˢ�³ɹ�</returns>
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

			// �����¼���������������Լ���ɫ���ݵ�ָ��
			Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints * patternNum];
			Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints * patternNum];
			int vertexBufferNumPoints = 0;

			for (int it = 0; it < numSparsePoints; ++it)
			{
				/* display modes:
				 * my_displayMode==0 - ��ʾ�ؼ�֡����������
				 * my_displayMode==1 - ��ʾ�ؼ�֡���������������������ͱ�Ե������
				 * my_displayMode==2 - ��ʾ�ؼ�֡�м�������
				 * my_displayMode==3 - ����ʾ�ؼ�֡����
				 */
				if (my_displayMode == 1 && originalInputSparse[it].status != InputPointStatus::STATUS_NORMAL
					&& originalInputSparse[it].status != InputPointStatus::STATUS_MARGINALIZE) continue;
				if (my_displayMode == 2 && originalInputSparse[it].status != InputPointStatus::STATUS_NORMAL) continue;
				if (my_displayMode > 2) continue;

				if (originalInputSparse[it].idpeth < 0) continue;

				float depth = 1.0f / originalInputSparse[it].idpeth;
				float depth4 = std::pow(depth, 4);
				float var = (1.0f / (originalInputSparse[it].idepth_hessian + 0.01));

				// TODO:���Ǳ���ϵ���������ֶ������depth2
				if (var * depth4 > my_scaledTH)
					continue;

				// ��������ȷ���ԽС������������Ⱦ���Խ��
				if (var > my_absTH)
					continue;

				// �������߳���Խ����ƥ���������ƥ����������ԽС
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

					// ����ʾģʽΪ��ʾ�ؼ�֡���������������趨��ͬ����ɫ��ʾ�ؼ�֡�в�ͬ��������
					// ����ʾģʽΪ��ʾ�ؼ�֡���Ѿ������Ҳ�δOutlier����������������ɫ����Ϊ���ػҶ�ֵ
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

			// ˢ�º����ˢ��ǰ�������ڴ�ռ��Լ��㹻�����������ݣ���ô�Ͳ���Ҫ���¿����ڴ�ռ�
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
		/// ���ƹؼ�֡��λ��
		/// </summary>
		/// <param name="lineWidth">���ƶ����߿�</param>
		/// <param name="color">���ƶ�����ɫ</param>
		/// <param name="sizeFactor">���ƶ����С����</param>
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
		/// ��ʾcolorBuffer��vertexBuffer����ʾ��������Ϣ
		/// </summary>
		/// <param name="pointSize">��������Ⱦ�ߴ�</param>
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