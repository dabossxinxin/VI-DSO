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
#include <string>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "util/settings.h"
#include "IOWrapper/ImageDisplay.h"
#include <opencv2/highgui/highgui.hpp>

namespace dso
{
	namespace IOWrap
	{
		std::unordered_set<std::string> openWindows;
		std::mutex openCVdisplayMutex;

		void displayImage(const char* windowName, const cv::Mat& image, bool autoSize)
		{
			if (setting_disableAllDisplay) return;

			std::unique_lock<std::mutex> lock(openCVdisplayMutex);
			if (!autoSize)
			{
				if (openWindows.find(windowName) == openWindows.end())
				{
					cv::namedWindow(windowName, cv::WINDOW_NORMAL);
					cv::resizeWindow(windowName, image.cols, image.rows);
					openWindows.insert(windowName);
				}
			}
			cv::imshow(windowName, image);
		}

		void displayImageStitch(const char* windowName, const std::vector<cv::Mat*> images, int cc, int rc)
		{
			if (setting_disableAllDisplay) return;
			if (images.empty()) return;

			// get dimensions.
			int w = images[0]->cols;
			int h = images[0]->rows;

			int num = std::max((int)setting_maxFrames, (int)images.size());

			// get optimal dimensions.
			int bestCC = 0;
			float bestLoss = 1e10;
			for (int cc = 1; cc < 10; cc++)
			{
				int ww = w * cc;
				int hh = h * ((num + cc - 1) / cc);

				float wLoss = ww / 16.0f;
				float hLoss = hh / 10.0f;
				float loss = std::max(wLoss, hLoss);

				if (loss < bestLoss)
				{
					bestLoss = loss;
					bestCC = cc;
				}
			}

			int bestRC = ((num + bestCC - 1) / bestCC);
			if (cc != 0)
			{
				bestCC = cc;
				bestRC = rc;
			}
			cv::Mat stitch = cv::Mat(bestRC * h, bestCC * w, images[0]->type());
			stitch.setTo(0);
			for (int i = 0; i < (int)images.size() && i < bestCC * bestRC; i++)
			{
				int c = i % bestCC;
				int r = i / bestCC;

				cv::Mat roi = stitch(cv::Rect(c * w, r * h, w, h));
				images[i]->copyTo(roi);
			}
			displayImage(windowName, stitch, false);
		}

		void displayImage(const char* windowName, const MinimalImageB* img, bool autoSize)
		{
			displayImage(windowName, cv::Mat(img->h, img->w, CV_8U, img->data), autoSize);
		}

		void displayImage(const char* windowName, const MinimalImageB3* img, bool autoSize)
		{
			displayImage(windowName, cv::Mat(img->h, img->w, CV_8UC3, img->data), autoSize);
		}

		void displayImage(const char* windowName, const MinimalImageF* img, bool autoSize)
		{
			displayImage(windowName, cv::Mat(img->h, img->w, CV_32F, img->data)*(1 / 254.0f), autoSize);
		}

		void displayImage(const char* windowName, const MinimalImageF3* img, bool autoSize)
		{
			displayImage(windowName, cv::Mat(img->h, img->w, CV_32FC3, img->data)*(1 / 254.0f), autoSize);
		}

		void displayImage(const char* windowName, const MinimalImageB16* img, bool autoSize)
		{
			displayImage(windowName, cv::Mat(img->h, img->w, CV_16U, img->data), autoSize);
		}

		void displayImageStitch(const char* windowName, const std::vector<MinimalImageB*> images, int cc, int rc)
		{
			std::vector<cv::Mat*> imagesCV;
			for (size_t i = 0; i < images.size(); i++)
				imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_8U, images[i]->data));
			displayImageStitch(windowName, imagesCV, cc, rc);
			for (size_t i = 0; i < images.size(); i++)
				delete imagesCV[i];
		}

		void displayImageStitch(const char* windowName, const std::vector<MinimalImageB3*> images, int cc, int rc)
		{
			// ��ͼ���ʽת��Ϊopencv�ĸ�ʽ����ʾ
			std::vector<cv::Mat*> imagesCV;
			for (size_t it = 0; it < images.size(); ++it)
				imagesCV.emplace_back(new cv::Mat(images[it]->h, images[it]->w, CV_8UC3, images[it]->data));
			displayImageStitch(windowName, imagesCV, cc, rc);
			for (size_t it = 0; it < images.size(); ++it)
				freePointer(imagesCV[it]);
		}

		void displayImageStitch(const char* windowName, const std::vector<MinimalImageF*> images, int cc, int rc)
		{
			std::vector<cv::Mat*> imagesCV;
			for (size_t i = 0; i < images.size(); i++)
				imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_32F, images[i]->data));
			displayImageStitch(windowName, imagesCV, cc, rc);
			for (size_t i = 0; i < images.size(); i++)
				delete imagesCV[i];
		}

		void displayImageStitch(const char* windowName, const std::vector<MinimalImageF3*> images, int cc, int rc)
		{
			std::vector<cv::Mat*> imagesCV;
			for (size_t i = 0; i < images.size(); i++)
				imagesCV.push_back(new cv::Mat(images[i]->h, images[i]->w, CV_32FC3, images[i]->data));
			displayImageStitch(windowName, imagesCV, cc, rc);
			for (size_t i = 0; i < images.size(); i++)
				delete imagesCV[i];
		}

		int waitKey(int milliseconds)
		{
			if (setting_disableAllDisplay) return 0;

			std::unique_lock<std::mutex> lock(openCVdisplayMutex);
			return cv::waitKey(milliseconds);
		}

		void closeAllWindows()
		{
			if (setting_disableAllDisplay) return;
			std::unique_lock<std::mutex> lock(openCVdisplayMutex);
			cv::destroyAllWindows();
			openWindows.clear();
		}
	}
}