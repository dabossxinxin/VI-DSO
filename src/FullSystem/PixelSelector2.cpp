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

#include <iostream>

#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

#include "IOWrapper/ImageDisplay.h"

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/PixelSelector2.h"

namespace dso
{
	/// <summary>
	/// 构造函数，生成必要的成员变量
	/// </summary>
	/// <param name="w">待提取特征图像的宽度</param>
	/// <param name="h">待提取特征图像的高度</param>
	PixelSelector::PixelSelector(int w, int h)
	{
		// 0xFF是十进制下的255
		randomPattern = new unsigned char[w * h];
		std::srand(3141592);
		for (int it = 0; it < w * h; ++it)
			randomPattern[it] = rand() & 0xFF;

		currentPotential = 3;

		int w32 = w / 32;
		int h32 = h / 32;
		int wh32 = w32 * h32;

		gradHist = new int[100 * (1 + w32) * (1 + h32)];
		ths = new float[wh32 + 100];
		thsSmoothed = new float[(wh32)+100];
		memset(thsSmoothed, 0, sizeof(float) * (wh32 + 100));

		allowFast = false;
		gradHistFrame = NULL;
	}

	/// <summary>
	/// 析构函数，析构主要成员变量
	/// </summary>
	PixelSelector::~PixelSelector()
	{
		freePointerVec(randomPattern);
		freePointerVec(gradHist);
		freePointerVec(ths);
		freePointerVec(thsSmoothed);
	}

	/// <summary>
	/// 根据梯度直方图信息以及需保留的特征数量计算梯度阈值
	/// </summary>
	/// <param name="hist">梯度直方图信息</param>
	/// <param name="below">按照这个比例去除像素点</param>
	/// <returns></returns>
	int computeHistQuantil(int* hist, float below)
	{
		int th = hist[0] * below + 0.5f;
		for (int it = 0; it < 90; ++it)
		{
			th -= hist[it + 1];
			if (th < 0) return it;
		}
		return 90;
	}

	/// <summary>
	/// 计算块状区域特征提取的梯度阈值：提取每个块状区域的梯度阈值并与领域做平均得到平滑后的阈值
	/// </summary>
	/// <param name="fh">待提取特征的图像帧</param>
	void PixelSelector::makeHists(const FrameHessian* const fh)
	{
		// 1、获取第0层金字塔图像的梯度数据
		gradHistFrame = fh;
		float* mapmax0 = fh->absSquaredGrad[0];

		int w = wG[0];
		int h = hG[0];

		int w32 = w / 32;
		int h32 = h / 32;
		thsStep = w32;

		// 2、遍历图像中的每一个块状区域，求解块状区域提取特征的梯度阈值
		for (int y = 0; y < h32; ++y)
		{
			for (int x = 0; x < w32; ++x)
			{
				float* map0 = mapmax0 + 32 * x + 32 * y * w;	// 获取块状区域头指针
				int* hist0 = gradHist;// + 50*(x+y*w32);
				memset(hist0, 0, sizeof(int) * 50);

				int colInImg = 0;
				int rowInImg = 0;
				int gradient = 0;

				// 遍历单个块状区域的像素点
				for (int rowInRoi = 0; rowInRoi < 32; ++rowInRoi)
				{
					for (int colInRoi = 0; colInRoi < 32; ++colInRoi)
					{
						colInImg = colInRoi + 32 * x;
						rowInImg = rowInRoi + 32 * y;
						if (colInImg > w - 2 || rowInImg > h - 2 ||
							colInImg < 1 || rowInImg < 1) continue;
						
						gradient = std::sqrtf(map0[colInRoi + rowInRoi * w]);
						if (gradient > 48) gradient = 48;
						hist0[gradient + 1]++;
						hist0[0]++;
					}
				}

				// 计算单个块状区域的梯度阈值
				ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
			}
		}

		// 3、平滑上述求解得到的每个块状区域的梯度阈值，方法是和领域的梯度阈值求平均
		for (int y = 0; y < h32; ++y)
		{
			for (int x = 0; x < w32; ++x)
			{
				float sum = 0;
				float num = 0;
				
				if (x > 0)
				{
					if (y > 0) { num++; sum += ths[x - 1 + (y - 1) * w32]; }
					if (y < h32 - 1) { num++; sum += ths[x - 1 + (y + 1) * w32]; }
					num++; sum += ths[x - 1 + (y)*w32];
				}

				if (x < w32 - 1)
				{
					if (y > 0) { num++; sum += ths[x + 1 + (y - 1) * w32]; }
					if (y < h32 - 1) { num++; sum += ths[x + 1 + (y + 1) * w32]; }
					num++; sum += ths[x + 1 + (y)*w32];
				}

				if (y > 0) { num++; sum += ths[x + (y - 1) * w32]; }
				if (y < h32 - 1) { num++; sum += ths[x + (y + 1) * w32]; }

				num++; sum += ths[x + y * w32];
				thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
			}
		}
	}

	/// <summary>
	/// 对于第0层金字塔图像，在其中选取特征
	/// step1：按照设定的pot大小在图像中选取特征
	/// step2：按照选取特征数量与期望特征数量关系动态调整pot大小并重新选点
	/// step3：递归一定次数后，若选点数量比期望数量还大太多，那么再筛选掉一些特征
	/// </summary>
	/// <param name="fh">待选择特征的帧</param>
	/// <param name="map_out">特征选择结果</param>
	/// <param name="density">期望选点数量</param>
	/// <param name="recursionsLeft">递归剩余次数</param>
	/// <param name="plot">是否绘制提取结果</param>
	/// <param name="thFactor">梯度阈值倍率</param>
	/// <returns>第0层金字塔图像中最终提取特征数量</returns>
	int PixelSelector::makeMaps(const FrameHessian* const fh, float* map_out,
		float density, int recursionsLeft, bool plot, float thFactor)
	{
		float numHave = 0;
		float numWant = density;
		float quotia;
		int idealPotential = currentPotential;

		{
			// the number of selected pixels behaves approximately as
			// K / (pot+1)^2, where K is a scene-dependent constant.
			// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

			// 计算块状区域梯度阈值
			if (fh != gradHistFrame) makeHists(fh);

			// 在第0层、第1层、第2层金字塔中选点
			Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

			numHave = n[0] + n[1] + n[2];
			quotia = numWant / numHave;

			// by default we want to over-sample by 40% just to be sure.
			float K = numHave * (currentPotential + 1) * (currentPotential + 1);
			idealPotential = sqrtf(K / numWant) - 1;
			if (idealPotential < 1) idealPotential = 1;

			// pot越大，选出的特征越少，pot越小，选出特征越多，通过动态调整pot大小递归选点
			if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1)
			{
				//re-sample to get more points!
				// potential needs to be smaller
				if (idealPotential >= currentPotential)
					idealPotential = currentPotential - 1;

				currentPotential = idealPotential;
				return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
			}
			else if (recursionsLeft > 0 && quotia < 0.25)
			{
				// re-sample to get less points!
				if (idealPotential <= currentPotential)
					idealPotential = currentPotential + 1;

				currentPotential = idealPotential;
				return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
			}
		}

		// 若实际选取到的特征还是比期望的多很多，那么再随机去除一些
		int numHaveSub = numHave;
		if (quotia < 0.95)
		{
			int wh = wG[0] * hG[0];
			int rn = 0;
			unsigned char charTH = 255 * quotia;
			for (int it = 0; it < wh; ++it)
			{
				if (map_out[it] != 0)
				{
					if (randomPattern[rn] > charTH)
					{
						map_out[it] = 0;
						numHaveSub--;
					}
					rn++;
				}
			}
		}

		currentPotential = idealPotential;

		if (plot)
		{
			int w = wG[0];
			int h = hG[0];
			int wh = w * h;

			// 绘制选取特征的图像
			float color = 0;
			MinimalImageB3 img(w, h);

			for (int it = 0; it < wh; ++it)
			{
				color = fh->dI[it][0] * 0.7;
				if (color > 255) color = 255;
				img.at(it) = Vec3b(color, color, color);
			}
			IOWrap::displayImage("Selector Image", &img);
			
#ifdef SAVE_INITIALIZER_DATA
			cv::Mat selImage = cv::Mat(img.h, img.w, CV_8UC3, img.data);
			std::string path0 = "./SelectorImage_lvl[0].png";
			cv::imwrite(path0, selImage);
#endif

			// 绘制选中的特征点
			int pixelIdx = 0;
			int featureNum = 0;
			for (int y = 0; y < h; ++y)
			{
				for (int x = 0; x < w; ++x)
				{
					pixelIdx = x + y * w;
					if (map_out[pixelIdx] == 1)
					{
						img.setPixelCirc(x, y, Vec3b(0, 255, 0));
						featureNum++;
					}
					else if (map_out[pixelIdx] == 2)
					{
						img.setPixelCirc(x, y, Vec3b(255, 0, 0));
						featureNum++;
					}
					else if (map_out[pixelIdx] == 4)
					{
						img.setPixelCirc(x, y, Vec3b(0, 0, 255));
						featureNum++;
					}   
				}
			}
			IOWrap::displayImage("Selector Pixels", &img);

#ifdef SAVE_INITIALIZER_DATA
			cv::Mat selPixel = cv::Mat(img.h, img.w, CV_8UC3, img.data);
			std::string path1 = "./SelectorPixel_feature["
				+ std::to_string(featureNum) + "]_lvl[0].png";
			cv::imwrite(path1, selPixel);
#endif
		}

		return numHaveSub;
	}

	/// <summary>
	/// 对于第零层金字塔图像提取特征，遍历每一个像素，并取在pot区域内梯度最大的像素
	/// </summary>
	/// <param name="fh">待提取特征的关键帧</param>
	/// <param name="map_out">特征提取结果</param>
	/// <param name="pot">在pot区域内提取满足阈值的最大梯度特征</param>
	/// <param name="thFactor">梯度阈值倍率</param>
	/// <returns>前三层金字塔中各自提取到的特征数量</returns>
	Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh, float* map_out, int pot, float thFactor)
	{
		// 获取图像前三层金字塔图像梯度信息
		Eigen::Vector3f const* const map0 = fh->dI;

		float* mapmax0 = fh->absSquaredGrad[0];
		float* mapmax1 = fh->absSquaredGrad[1];
		float* mapmax2 = fh->absSquaredGrad[2];

		int w = wG[0];
		int w1 = wG[1];
		int w2 = wG[2];
		int h = hG[0];
		
		// 0~PI等分为16等份，每份角度间隔为11.5度
		const Vec2f directions[16] = {
				 Vec2f(0,		  1.0000),
				 Vec2f(0.3827,    0.9239),
				 Vec2f(0.1951,    0.9808),
				 Vec2f(0.9239,    0.3827),
				 Vec2f(0.7071,    0.7071),
				 Vec2f(0.3827,   -0.9239),
				 Vec2f(0.8315,    0.5556),
				 Vec2f(0.8315,   -0.5556),
				 Vec2f(0.5556,   -0.8315),
				 Vec2f(0.9808,    0.1951),
				 Vec2f(0.9239,   -0.3827),
				 Vec2f(0.7071,   -0.7071),
				 Vec2f(0.5556,    0.8315),
				 Vec2f(0.9808,   -0.1951),
				 Vec2f(1.0000,    0.0000),
				 Vec2f(0.1951,   -0.9808) };

		memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

		float dw1 = setting_gradDownweightPerLevel;
		float dw2 = dw1 * dw1;

		int n3 = 0, n2 = 0, n4 = 0;
		for (int y4 = 0; y4 < h; y4 += (4 * pot)) for (int x4 = 0; x4 < w; x4 += (4 * pot))
		{
			int my3 = std::min((4 * pot), h - y4);
			int mx3 = std::min((4 * pot), w - x4);
			int bestIdx4 = -1; float bestVal4 = 0;
			Vec2f dir4 = directions[randomPattern[n2] & 0xF];
			for (int y3 = 0; y3 < my3; y3 += (2 * pot)) for (int x3 = 0; x3 < mx3; x3 += (2 * pot))
			{
				int x34 = x3 + x4;
				int y34 = y3 + y4;
				int my2 = std::min((2 * pot), h - y34);
				int mx2 = std::min((2 * pot), w - x34);
				int bestIdx3 = -1; float bestVal3 = 0;
				Vec2f dir3 = directions[randomPattern[n2] & 0xF];
				for (int y2 = 0; y2 < my2; y2 += pot) for (int x2 = 0; x2 < mx2; x2 += pot)
				{
					int x234 = x2 + x34;
					int y234 = y2 + y34;
					int my1 = std::min(pot, h - y234);
					int mx1 = std::min(pot, w - x234);
					int bestIdx2 = -1; float bestVal2 = 0;
					Vec2f dir2 = directions[randomPattern[n2] & 0xF];
					for (int y1 = 0; y1 < my1; y1 += 1) for (int x1 = 0; x1 < mx1; x1 += 1)
					{
						assert(x1 + x234 < w);
						assert(y1 + y234 < h);
						int xf = x1 + x234;
						int yf = y1 + y234;
						int idx = xf + w * yf;

						if (xf < 4 || xf > w - 4 || yf < 4 || yf > h - 4) continue;

						float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];	// 零层金字塔阈值
						float pixelTH1 = pixelTH0 * dw1;								// 一层金字塔阈值
						float pixelTH2 = pixelTH1 * dw2;								// 二层金字塔阈值

						float ag0 = mapmax0[idx];
						if (ag0 > pixelTH0 * thFactor)
						{
							Vec2f ag0d = map0[idx].tail<2>();
							float dirNorm = fabsf((float)(ag0d.dot(dir2)));				// 计算梯度在随机方向上的分量
							if (!setting_selectDirectionDistribution) dirNorm = ag0;

							if (dirNorm > bestVal2)
							{
								bestVal2 = dirNorm; 
								bestIdx2 = idx; 
								bestIdx3 = -2; 
								bestIdx4 = -2;
							}
						}
						if (bestIdx3 == -2) continue;

						float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) + (int)(yf * 0.5f + 0.25f) * w1];
						if (ag1 > pixelTH1 * thFactor)
						{
							Vec2f ag0d = map0[idx].tail<2>();
							float dirNorm = fabsf((float)(ag0d.dot(dir3)));
							if (!setting_selectDirectionDistribution) dirNorm = ag1;

							if (dirNorm > bestVal3)
							{
								bestVal3 = dirNorm; 
								bestIdx3 = idx; 
								bestIdx4 = -2;
							}
						}
						if (bestIdx4 == -2) continue;

						float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) + (int)(yf * 0.25f + 0.125) * w2];
						if (ag2 > pixelTH2 * thFactor)
						{
							Vec2f ag0d = map0[idx].tail<2>();
							float dirNorm = fabsf((float)(ag0d.dot(dir4)));
							if (!setting_selectDirectionDistribution) dirNorm = ag2;

							if (dirNorm > bestVal4)
							{
								bestVal4 = dirNorm; 
								bestIdx4 = idx;
							}
						}
					}

					// 在第0层金字塔上找到的特征
					if (bestIdx2 > 0)
					{
						map_out[bestIdx2] = 1;
						bestVal3 = 1e10;
						n2++;
					}
				}

				// 在第1层金字塔上找到的特征
				if (bestIdx3 > 0)
				{
					map_out[bestIdx3] = 2;
					bestVal4 = 1e10;
					n3++;
				}
			}

			// 在第2层金字塔上找到的特征
			if (bestIdx4 > 0)
			{
				map_out[bestIdx4] = 4;
				n4++;
			}
		}

		return Eigen::Vector3i(n2, n3, n4);
	}
}