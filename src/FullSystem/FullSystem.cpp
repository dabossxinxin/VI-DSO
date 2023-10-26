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
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/FullSystem.h"
#include "FullSystem/ResidualProjections.h"

namespace dso
{
	int FrameHessian::instanceCounter = 0;
	int PointHessian::instanceCounter = 0;
	int CalibHessian::instanceCounter = 0;

	void quaternionNormalize(Eigen::Vector4d& q)
	{
		q = q / q.norm();
	}

	Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R) \
	{
		Eigen::Vector4d q = Eigen::Vector4d::Zero();
		double trR = R(0, 0) + R(1, 1) + R(2, 2);
		q(3) = sqrt(trR + 1) / 2;
		q(0) = (R(2, 1) - R(1, 2)) / 4 / q(3);
		q(1) = (R(0, 2) - R(2, 0)) / 4 / q(3);
		q(2) = (R(1, 0) - R(0, 1)) / 4 / q(3);
		quaternionNormalize(q);
		return q;
	}

	/// <summary>
	/// FullSystem构造函数：为主要成员变量赋初值
	/// </summary>
	FullSystem::FullSystem()
	{
		int retstat = 0;
		if (setting_logStuff)
		{
			retstat += system("rm -rf logs");
			retstat += system("mkdir logs");

			retstat += system("rm -rf mats");
			retstat += system("mkdir mats");

			calibLog = new std::ofstream();
			calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
			calibLog->precision(12);

			numsLog = new std::ofstream();
			numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
			numsLog->precision(10);

			coarseTrackingLog = new std::ofstream();
			coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
			coarseTrackingLog->precision(10);

			eigenAllLog = new std::ofstream();
			eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
			eigenAllLog->precision(10);

			eigenPLog = new std::ofstream();
			eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
			eigenPLog->precision(10);

			eigenALog = new std::ofstream();
			eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
			eigenALog->precision(10);

			DiagonalLog = new std::ofstream();
			DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
			DiagonalLog->precision(10);

			variancesLog = new std::ofstream();
			variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
			variancesLog->precision(10);

			nullspacesLog = new std::ofstream();
			nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
			nullspacesLog->precision(10);
		}
		else
		{
			nullspacesLog = NULL;
			variancesLog = NULL;
			DiagonalLog = NULL;
			eigenALog = NULL;
			eigenPLog = NULL;
			eigenAllLog = NULL;
			numsLog = NULL;
			calibLog = NULL;
		}

		assert(retstat != 293847);

		selectionMap = new float[wG[0] * hG[0]];

		coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
		coarseTracker = new CoarseTracker(wG[0], hG[0]);
		coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
		coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
		pixelSelector = new PixelSelector(wG[0], hG[0]);

		statistics_lastNumOptIts = 0;
		statistics_numDroppedPoints = 0;
		statistics_numActivatedPoints = 0;
		statistics_numCreatedPoints = 0;
		statistics_numForceDroppedResBwd = 0;
		statistics_numForceDroppedResFwd = 0;
		statistics_numMargResFwd = 0;
		statistics_numMargResBwd = 0;

		lastCoarseRMSE.setConstant(100);

		currentMinActDist = 2;
		initialized = false;

		ef = new EnergyFunctional();
		ef->red = &this->treadReduce;

		isLost = false;
		initFailed = false;
		linearizeOperation = true;
		runMapping = true;

		needNewKFAfter = -1;
		lastRefStopID = 0;
		minIdJetVisDebug = -1;
		maxIdJetVisDebug = -1;
		minIdJetVisTracker = -1;
		maxIdJetVisTracker = -1;

		mappingThread = std::thread(&FullSystem::mappingLoop, this);
	}

	/// <summary>
	/// FullSystem析构函数
	/// </summary>
	FullSystem::~FullSystem()
	{
		// 1、等待mapping线程完成
		blockUntilMappingIsFinished();

		// 2、释放成员变量的内存空间
		if (setting_logStuff)
		{
			calibLog->close(); delete calibLog;
			numsLog->close(); delete numsLog;
			coarseTrackingLog->close(); delete coarseTrackingLog;
			errorsLog->close(); delete errorsLog;
			eigenAllLog->close(); delete eigenAllLog;
			eigenPLog->close(); delete eigenPLog;
			eigenALog->close(); delete eigenALog;
			DiagonalLog->close(); delete DiagonalLog;
			variancesLog->close(); delete variancesLog;
			nullspacesLog->close(); delete nullspacesLog;
		}

		delete[] selectionMap;

		for (FrameShell* s : allFrameHistory)
			delete s;
		for (FrameHessian* fh : unmappedTrackedFrames)
			delete fh;

		delete coarseDistanceMap;
		delete coarseTracker;
		delete coarseTracker_forNewKF;
		delete coarseInitializer;
		delete pixelSelector;
		delete ef;
	}

	/// <summary>
	/// 未实现函数
	/// </summary>
	/// <param name="originalCalib"></param>
	/// <param name="originalW"></param>
	/// <param name="originalH"></param>
	void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
	{

	}

	/// <summary>
	/// 设置Gamma校正参数
	/// </summary>
	/// <param name="BInv">校正参数</param>
	void FullSystem::setGammaFunction(float* BInv)
	{
		if (BInv == NULL) return;

		// copy BInv.
		memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

		// invert.
		for (int it = 1; it < 255; ++it)
		{
			// find val, such that Binv[val] = i.
			// I dont care about speed for this, so do it the stupid way.
			for (int s = 1; s < 255; s++)
			{
				if (BInv[s] <= it && BInv[s + 1] >= it)
				{
					Hcalib.B[it] = s + (it - BInv[s]) / (BInv[s + 1] - BInv[s]);
					break;
				}
			}
		}
		Hcalib.B[0] = 0;
		Hcalib.B[255] = 255;
	}

	/// <summary>
	/// 输出系统中所有帧的位姿信息
	/// </summary>
	/// <param name="file">信息保存路径</param>
	void FullSystem::printResult(std::string file)
	{
		std::unique_lock<std::mutex> lock(trackMutex);
		std::unique_lock<std::mutex> crlock(shellPoseMutex);

		std::ofstream myfile;
		myfile.open(file.c_str());
		myfile << std::setprecision(15);

		for (FrameShell* s : allFrameHistory)
		{
			if (!s->poseValid) continue;

			if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

			myfile << s->timestamp <<
				" " << s->camToWorld.translation().transpose() <<
				" " << s->camToWorld.so3().unit_quaternion().x() <<
				" " << s->camToWorld.so3().unit_quaternion().y() <<
				" " << s->camToWorld.so3().unit_quaternion().z() <<
				" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		}
		myfile.close();
	}

	/// <summary>
	/// 系统跟踪最新帧的位姿
	/// </summary>
	/// <param name="fh">最新进入系统的帧</param>
	/// <returns>
	/// [0]：跟踪时第零层金字塔的光度残差；
	/// [1]：跟踪时第零层金字塔仅平移时参考帧与最新帧间的像素位移；
	/// [2]：固定为0；
	/// [3]：跟踪时第零层金字塔平移与旋转时参考帧与最新帧间的像素位移；
	/// </returns>
	Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
	{
		assert(allFrameHistory.size() > 0);

		for (IOWrap::Output3DWrapper* ow : outputWrapper)
			ow->pushLiveFrame(fh);

		FrameHessian* lastF = coarseTracker->lastRef;
		AffLight aff_last_2_l = AffLight(0, 0);
		std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

		// 1、根据既定规则设置一系列最新帧与参考关键帧之间的位姿变换
		if (use_stereo && (allFrameHistory.size() == 2 || first_track_flag == false))
		{
			//initializeFromInitializer(fh);
			first_track_flag = true;
			lastF_2_fh_tries.emplace_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double, 3, 1>::Zero()));

			for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
			{
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));
				lastF_2_fh_tries.emplace_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));
			}

			coarseTracker->makeK(&Hcalib);
			coarseTracker->setCTRefForFirstFrame(frameHessians);
			lastF = coarseTracker->lastRef;
		}
		else if (allFrameHistory.size() == 2) 
		{
			for (unsigned int it = 0; it < lastF_2_fh_tries.size(); ++it)
				lastF_2_fh_tries.emplace_back(SE3());
		}
		else
		{
			FrameShell* slast = allFrameHistory[allFrameHistory.size() - 2];		// 最新帧上一帧
			FrameShell* sprelast = allFrameHistory[allFrameHistory.size() - 3];		// 最新帧上上帧
			SE3 slast_2_sprelast;
			SE3 lastF_2_slast;

			{
				std::unique_lock<std::mutex> crlock(shellPoseMutex);
				slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
				lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
				aff_last_2_l = slast->aff_g2l;
			}
			SE3 fh_2_slast = slast_2_sprelast;

			lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast);						// 匀速模型
			lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// 倍速模型
			lastF_2_fh_tries.emplace_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast);	// 半速模型
			lastF_2_fh_tries.emplace_back(lastF_2_slast);												// 零速模型
			lastF_2_fh_tries.emplace_back(SE3());														// 静止模型

			// just try a TON of different initializations (all rotations). In the end,
			// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
			// also, if tracking rails here we loose, so we really, really want to avoid that.
			for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.02)
			{
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			}

			if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
			{
				lastF_2_fh_tries.clear();
				lastF_2_fh_tries.emplace_back(SE3());
			}
		}

		Vec3 flowVecs = Vec3(100, 100, 100);
		SE3 lastF_2_fh = SE3();
		AffLight aff_g2l = AffLight(0, 0);

		// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
		// I'll keep track of the so-far best achieved residual for each level in achievedRes.
		// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
		Vec5 achievedRes = Vec5::Constant(NAN);
		bool haveOneGood = false;
		int tryIterations = 0;

		// 2、一一尝试上述设置的一系类位姿变换，计算出光度残差较小的尝试作为跟踪结果
		for (unsigned int it = 0; it < lastF_2_fh_tries.size(); ++it)
		{
			if (use_stereo && frameHessians.size() < setting_maxFrames - 1)
			{
				if (it > 0)
				{
					initFailed = true;
					first_track_flag = false;
				}
			}

			AffLight aff_g2l_this = aff_last_2_l;
			SE3 lastF_2_fh_this = lastF_2_fh_tries[it];
			bool trackingIsGood = coarseTracker->trackNewestCoarse(fh, lastF_2_fh_this,
				aff_g2l_this, pyrLevelsUsed - 1, achievedRes);
			tryIterations++;

			if (it != 0)
			{
				printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					it,
					it, pyrLevelsUsed - 1,
					aff_g2l_this.a, aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
			}

			// 若跟踪成功并且第零层光度残差小于所记录的第零层最小残差，将此时的位姿及光度变换视为最优
			if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) &&
				!(coarseTracker->lastResiduals[0] >= achievedRes[0]))
			{
				flowVecs = coarseTracker->lastFlowIndicators;
				aff_g2l = aff_g2l_this;
				lastF_2_fh = lastF_2_fh_this;
				haveOneGood = true;
			}

			// 本次尝试跟踪成功，更新各层金字塔的最小光度残差阈值
			if (haveOneGood)
			{
				for (int idx = 0; idx < 5; ++idx)
				{
					if (!std::isfinite((float)achievedRes[idx]) || achievedRes[idx] > coarseTracker->lastResiduals[idx])
						achievedRes[idx] = coarseTracker->lastResiduals[idx];
				}
			}

			// 若本次尝试的光度残差明显很小，那么直接认为本次尝试的效果很好，可直接退出循环
			if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
				break;
		}

		// 3、若找不到光度残差较低的尝试那么认为跟踪失败
		if (!haveOneGood)
		{
			printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
			flowVecs = Vec3(0, 0, 0);
			aff_g2l = aff_last_2_l;
			lastF_2_fh = lastF_2_fh_tries[0];
		}

		lastCoarseRMSE = achievedRes;

		// 4、记录跟踪结果：包含当前帧位姿以及光度参数
		fh->shell->camToTrackingRef = lastF_2_fh.inverse();
		fh->shell->trackingRef = lastF->shell;
		fh->shell->aff_g2l = aff_g2l;
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

		if (coarseTracker->firstCoarseRMSE < 0)
			coarseTracker->firstCoarseRMSE = achievedRes[0];

		if (!setting_debugout_runquiet)
			printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

		if (setting_logStuff)
		{
			(*coarseTrackingLog) << std::setprecision(16)
				<< fh->shell->id << " "
				<< fh->shell->timestamp << " "
				<< fh->ab_exposure << " "
				<< fh->shell->camToWorld.log().transpose() << " "
				<< aff_g2l.a << " "
				<< aff_g2l.b << " "
				<< achievedRes[0] << " "
				<< tryIterations << "\n";
		}

		return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
	}

	/// <summary>
	/// 将关键帧中的未激活点在最新帧中跟踪一遍，优化其逆深度信息
	/// </summary>
	/// <param name="fh">系统中的最新帧</param>
	void FullSystem::traceNewCoarse(FrameHessian* fh)
	{
		std::unique_lock<std::mutex> lock(mapMutex);

		int trace_total = 0;
		int trace_good = 0;
		int trace_oob = 0;
		int trace_out = 0;
		int trace_skip = 0;
		int trace_badcondition = 0;
		int trace_uninitialized = 0;

		Mat33f K = Mat33f::Identity();
		K(0, 0) = Hcalib.fxl();
		K(1, 1) = Hcalib.fyl();
		K(0, 2) = Hcalib.cxl();
		K(1, 2) = Hcalib.cyl();

		for (FrameHessian* host : frameHessians)
		{
			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			Vec3f Kt = K * hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			// host帧中的未激活点在最新帧中跟踪一遍，优化点的状态
			for (ImmaturePoint* ph : host->immaturePoints)
			{
				ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}
	}

	// process nonkey frame to refine key frame idepth
	void FullSystem::traceNewCoarseNonKey(FrameHessian *fh, FrameHessian *fhRight)
	{
		std::unique_lock<std::mutex> lock(mapMutex);

		float idepth_min_update = 0;
		float idepth_max_update = 0;

		Mat33f K = Mat33f::Identity();
		K(0, 0) = Hcalib.fxl();
		K(1, 1) = Hcalib.fyl();
		K(0, 2) = Hcalib.cxl();
		K(1, 2) = Hcalib.cyl();
		Mat33f Ki = K.inverse();

		for (FrameHessian *host : frameHessians)
		{
			int trace_total = 0;
			int	trace_good = 0;
			int	trace_oob = 0;
			int	trace_out = 0;
			int trace_skip = 0;
			int	trace_badcondition = 0;
			int	trace_uninitialized = 0;

			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
			Vec3f Kt = K * hostToNew.translation().cast<float>();
			Vec3f t = hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			for (ImmaturePoint *ph : host->immaturePoints)
			{
				// do temperol stereo match
				auto phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);
				if (phTrackStatus == ImmaturePointStatus::IPS_GOOD)
				{
					ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

					Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
					float idepth_min_project = 1.0f / ptpMin[2];
					Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
					float idepth_max_project = 1.0f / ptpMax[2];

					phNonKey->idepth_min = idepth_min_project;
					phNonKey->idepth_max = idepth_max_project;
					phNonKey->u_stereo = phNonKey->u;
					phNonKey->v_stereo = phNonKey->v;
					phNonKey->idepth_min_stereo = phNonKey->idepth_min;
					phNonKey->idepth_max_stereo = phNonKey->idepth_max;

					// 双目匹配
					auto phNonKeyStereoStatus = phNonKey->traceStereo(fhRight, &Hcalib, 1);
					if (phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD)
					{
						ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0),
							phNonKey->lastTraceUV(1), fhRight, &Hcalib);

						phNonKeyRight->u_stereo = phNonKeyRight->u;
						phNonKeyRight->v_stereo = phNonKeyRight->v;
						phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
						phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

						// do static stereo match from right image to left
						ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, &Hcalib, 0);

						// change of u after two different stereo match
						float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
						float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

						// free to debug the threshold
						if (u_stereo_delta > 1 && disparity < 10)
						{
							ph->lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
							continue;
						}
						else
						{
							// project back
							Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
							idepth_min_update = 1.0f / pinverse_min(2);

							Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
							idepth_max_update = 1.0f / pinverse_max(2);

							ph->idepth_min = idepth_min_update;
							ph->idepth_max = idepth_max_update;

							delete phNonKey;
							delete phNonKeyRight;
						}
					}
					else
					{
						delete phNonKey;
						continue;
					}
				}
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}

		}
	}

	void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fhRight)
	{
		std::unique_lock<std::mutex> lock(mapMutex);

		int trace_total = 0;
		int	trace_good = 0;
		int	trace_oob = 0;
		int	trace_out = 0;
		int	trace_skip = 0;
		int	trace_badcondition = 0;
		int	trace_uninitialized = 0;

		Mat33f K = Mat33f::Identity();
		K(0, 0) = Hcalib.fxl();
		K(1, 1) = Hcalib.fyl();
		K(0, 2) = Hcalib.cxl();
		K(1, 2) = Hcalib.cyl();

		for (FrameHessian* host : frameHessians)
		{
			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			Vec3f Kt = K * hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			for (ImmaturePoint* ph : host->immaturePoints)
			{
				ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}
	}

	/// <summary>
	/// 激活未成熟特征
	/// </summary>
	/// <param name="optimized">优化后的特征</param>
	/// <param name="toOptimize">待激活特征</param>
	/// <param name="min">待激活点索引</param>
	/// <param name="max">待激活点索引</param>
	/// <param name="stats">状态返回值，该函数中没作用</param>
	/// <param name="tid">线程ID</param>
	void FullSystem::activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize, int min, int max, Vec10* stats, int tid)
	{
		ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
		for (int k = min; k < max; ++k)
		{
			(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
		}
		delete[] tr;
	}

	/// <summary>
	/// 一次跟踪成功后，对关键帧中的未激活点进行优化并激活；
	/// step1：根据系统参数设定计算特征稀疏参数currentMinActDis；
	/// step2：计算滑窗关键帧中最新帧中的特征距离地图便于均匀选取未激活特征；
	/// step3：在滑窗关键帧中选择满足激活条件以及均匀性条件的特征准备优化激活；
	/// step4：针对选定的特征进行基于多帧观测的优化，并将优化结果写入关键帧数据中；
	/// </summary>
	void FullSystem::activatePointsMT()
	{
		if (ef->nPoints < setting_desiredPointDensity*0.66)
			currentMinActDist -= 0.8;
		if (ef->nPoints < setting_desiredPointDensity*0.8)
			currentMinActDist -= 0.5;
		else if (ef->nPoints < setting_desiredPointDensity*0.9)
			currentMinActDist -= 0.2;
		else if (ef->nPoints < setting_desiredPointDensity)
			currentMinActDist -= 0.1;

		if (ef->nPoints > setting_desiredPointDensity*1.5)
			currentMinActDist += 0.8;
		if (ef->nPoints > setting_desiredPointDensity*1.3)
			currentMinActDist += 0.5;
		if (ef->nPoints > setting_desiredPointDensity*1.15)
			currentMinActDist += 0.2;
		if (ef->nPoints > setting_desiredPointDensity)
			currentMinActDist += 0.1;

		if (currentMinActDist < 0) currentMinActDist = 0;
		if (currentMinActDist > 4) currentMinActDist = 4;

		if (!setting_debugout_runquiet)
			printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
				currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

		FrameHessian* newestHs = frameHessians.back();

		// 1、以滑窗中的最新帧构造距离地图，便于均匀选取未激活点进行优化激活操作
		coarseDistanceMap->makeK(&Hcalib);
		coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

		//coarseTracker->debugPlotDistMap("distMap");

		std::vector<ImmaturePoint*> toOptimize; 
		toOptimize.reserve(20000);

		// 2、遍历滑窗中所有关键帧，选择能被激活的特征加入toOptimized中准备进行优化激活
		for (FrameHessian* host : frameHessians)
		{
			if (host == newestHs) continue;

			SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
			Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

			for (unsigned int it = 0; it < host->immaturePoints.size(); ++it)
			{
				auto ph = host->immaturePoints[it];
				ph->idxInImmaturePoints = it;

				if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
				{
					delete ph;
					host->immaturePoints[it] = NULL;
					continue;
				}

				bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB)
					&& ph->lastTracePixelInterval < 8
					&& ph->quality > setting_minTraceQuality
					&& (ph->idepth_max + ph->idepth_min) > 0;

				if (!canActivate)
				{
					if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
					{
						delete ph;
						host->immaturePoints[it] = NULL;
					}
					continue;
				}

				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f*(ph->idepth_max + ph->idepth_min));
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				if ((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
				{
					float dist = (ptp[0] - floorf((float)(ptp[0]))) +
						float(coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v]);

					if (dist >= currentMinActDist * ph->my_type)
					{
						coarseDistanceMap->addIntoDistFinal(u, v);
						toOptimize.emplace_back(ph);
					}
				}
				else
				{
					delete ph;
					host->immaturePoints[it] = NULL;
				}
			}
		}

		// 3、特征根据多帧观测以及观测的光度残差进行优化，并将优化结果写入滑窗关键帧中
		std::vector<PointHessian*> optimized; 
		optimized.resize(toOptimize.size());

		if (multiThreading)
			treadReduce.reduce(std::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize,
				std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 0, toOptimize.size(), 50);
		else
			activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

		for (unsigned k = 0; k < toOptimize.size(); ++k)
		{
			PointHessian* opt = optimized[k];
			ImmaturePoint* ph = toOptimize[k];

			if (opt != 0 && opt != (PointHessian*)((long)(-1)))
			{
				opt->host->immaturePoints[ph->idxInImmaturePoints] = NULL;
				opt->host->pointHessians.emplace_back(opt);
				ef->insertPoint(opt);
				for (PointFrameResidual* r : opt->residuals)
					ef->insertResidual(r);
				assert(opt->efPoint != NULL);
				delete ph;
			}
			else if (opt == (PointHessian*)((long)(-1)) || ph->lastTraceStatus == IPS_OOB)
			{
				ph->host->immaturePoints[ph->idxInImmaturePoints] = NULL;
				delete ph;
			}
			else
			{
				assert(opt == 0 || opt == (PointHessian*)((long)(-1)));
			}
		}

		// 4、清除滑窗关键帧中未激活特征序列中数据为空的部分
		for (FrameHessian* host : frameHessians)
		{
			for (int it = 0; it < (int)host->immaturePoints.size(); ++it)
			{
				if (host->immaturePoints[it] == NULL)
				{
					host->immaturePoints[it] = host->immaturePoints.back();
					host->immaturePoints.pop_back();
					it--;
				}
			}
		}
	}

	/// <summary>
	/// 对于已经激活的特征去掉其中质量不好的特征
	/// </summary>
	void FullSystem::flagPointsForRemoval()
	{
		assert(EFIndicesValid);

		std::vector<FrameHessian*> fhsToKeepPoints;
		std::vector<FrameHessian*> fhsToMargPoints;

		//if(setting_margPointVisWindow>0)
		{
			for (int i = ((int)frameHessians.size()) - 1; i >= 0 && i >= ((int)frameHessians.size()); i--)
				if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

			for (int i = 0; i < (int)frameHessians.size(); i++)
				if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
		}

		//ef->setAdjointsF();
		//ef->setDeltaF(&Hcalib);
		int flag_oob = 0;
		int flag_in = 0;
		int flag_inin = 0;
		int flag_ignores = 0;

		for (FrameHessian* host : frameHessians)
		{
			for (unsigned int it = 0; it < host->pointHessians.size(); ++it)
			{
				auto ph = host->pointHessians[it];
				if (ph == NULL) continue;

				if (ph->idepth_scaled < 0 || ph->residuals.empty())
				{
					host->pointHessiansOut.emplace_back(ph);
					host->pointHessians[it] = NULL;
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					flag_ignores++;
				}
				else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
				{
					flag_oob++;
					if (ph->isInlierNew())
					{
						flag_in++;
						int ngoodRes = 0;
						for (PointFrameResidual* r : ph->residuals)
						{
							r->resetOOB();
							if (r->stereoResidualFlag)
								r->linearizeStereo(&Hcalib);
							else
								r->linearize(&Hcalib);
							r->efResidual->isLinearized = false;
							r->applyRes(true);
							if (r->efResidual->isActive())
							{
								r->efResidual->fixLinearizationF(ef);
								ngoodRes++;
							}
						}
						if (ph->idepth_hessian > setting_minIdepthH_marg)
						{
							flag_inin++;
							ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
							host->pointHessiansMarginalized.emplace_back(ph);
						}
						else
						{
							ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
							host->pointHessiansOut.emplace_back(ph);
						}
					}
					else
					{
						host->pointHessiansOut.emplace_back(ph);
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					}

					host->pointHessians[it] = NULL;
				}
			}

			// 去掉pointHessians中被声明为空指针的特征数据
			for (int it = 0; it < (int)host->pointHessians.size(); ++it)
			{
				if (host->pointHessians[it] == NULL)
				{
					host->pointHessians[it] = host->pointHessians.back();
					host->pointHessians.pop_back();
					it--;
				}
			}
		}
	}

	/// <summary>
	/// 根据Initializer handle的结果进行初始化操作；
	/// step1：统计初始化handle中得到的特征逆深度并计算逆深度归一化因子；
	/// step2：将初始化handle中得到的特征加入以firstFrame为主帧的激活特征中
	/// step3：使用归一化因子归一化帧间位姿平移参数并将该位姿以及光度参数赋值给初始化帧
	/// </summary>
	/// <param name="newFrame">DSO系统最新进入的帧</param>
	void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
	{
		std::unique_lock<std::mutex> lock(mapMutex);

		coarseInitializer->firstFrame->idx = frameHessians.size();
		frameHessians.emplace_back(coarseInitializer->firstFrame);
		coarseInitializer->firstFrame->frameID = allKeyFramesHistory.size();
		coarseInitializer->firstFrame->frameRight->frameID = 10000 + allKeyFramesHistory.size();
		allKeyFramesHistory.emplace_back(coarseInitializer->firstFrame->shell);

		ef->insertFrame(coarseInitializer->firstFrame, &Hcalib);
		setPrecalcValues();
		FrameHessian* firstFrameRight = coarseInitializer->firstFrameRight;

		//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
		//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

		coarseInitializer->firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
		coarseInitializer->firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
		coarseInitializer->firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

		float idepthStereo = 0;
		float sumIDepth = 1e-5;
		float numIDepth = 1e-5;

		// 1、统计初始化器中计算得到的特征逆深度，并计算逆深度归一化因子
		for (int it = 0; it < coarseInitializer->numPoints[0]; ++it)
		{
			sumIDepth += coarseInitializer->points[0][it].iR;
			numIDepth++;
		}
		float rescaleFactor = 1 / (sumIDepth / numIDepth);

		float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];
		if (!setting_debugout_runquiet)
			printf("Initialization: keep %.1f%% (need %d, have %d)!\n",
				100 * keepPercentage, (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0]);

		// 2、将初始化器中计算得到的特征加入到以firstFrame为主帧的激活点中并添加到energyFunction中
		if (use_stereo)
		{
			for (int idx = 0; idx < coarseInitializer->numPoints[0]; ++idx)
			{
				if (rand() / (float)RAND_MAX > keepPercentage) continue;

				Pnt* point = coarseInitializer->points[0] + idx;
				ImmaturePoint* pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f,
					coarseInitializer->firstFrame, point->my_type, &Hcalib);

				pt->u_stereo = pt->u;
				pt->v_stereo = pt->v;
				pt->idepth_min_stereo = 0;
				pt->idepth_max_stereo = NAN;

				pt->traceStereo(firstFrameRight, &Hcalib, 1);

				pt->idepth_min = pt->idepth_min_stereo;
				pt->idepth_max = pt->idepth_max_stereo;
				idepthStereo = pt->idepth_stereo;

				if (!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) ||
					!std::isfinite(pt->idepth_max) || pt->idepth_min < 0 || pt->idepth_max < 0)
				{
					delete pt;
					pt = NULL;
					continue;
				}

				PointHessian* ph = new PointHessian(pt, &Hcalib);
				if (pt != NULL) 
				{ 
					delete pt; 
					pt = NULL;
					pt = NULL; 
				}
				if (!std::isfinite(ph->energyTH)) 
				{ 
					delete ph; 
					ph = NULL;
					continue; 
				}

				ph->setIdepthScaled(idepthStereo);
				ph->setIdepthZero(idepthStereo);
				ph->hasDepthPrior = true;
				ph->setPointStatus(PointHessian::ACTIVE);

				coarseInitializer->firstFrame->pointHessians.emplace_back(ph);
				ef->insertPoint(ph);

				PointFrameResidual* r = new PointFrameResidual(ph, ph->host, ph->host->frameRight);
				r->state_NewEnergy = r->state_energy = 0;
				r->state_NewState = ResState::OUTLIER;
				r->setState(ResState::INNER);
				r->stereoResidualFlag = true;
				ph->residuals.emplace_back(r);
				ef->insertResidual(r);
			}
		}
		else
		{
			for (int idx = 0; idx < coarseInitializer->numPoints[0]; ++idx)
			{
				if (rand() / (float)RAND_MAX > keepPercentage) continue;

				Pnt* point = coarseInitializer->points[0] + idx;
				ImmaturePoint* pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f,
					coarseInitializer->firstFrame, point->my_type, &Hcalib);
				if (!std::isfinite(pt->energyTH)) 
				{ 
					delete pt;
					pt = NULL;
					continue;
				}

				pt->idepth_max = pt->idepth_min = 1;
				PointHessian* ph = new PointHessian(pt, &Hcalib);
				if (pt != NULL) 
				{ 
					delete pt; 
					pt = NULL; 
				}
				if (!std::isfinite(ph->energyTH)) 
				{ 
					delete ph; 
					ph = NULL;
					continue; 
				}

				ph->setIdepthScaled(point->iR* rescaleFactor);
				ph->setIdepthZero(ph->idepth);
				ph->hasDepthPrior = true;
				ph->setPointStatus(PointHessian::ACTIVE);

				coarseInitializer->firstFrame->pointHessians.emplace_back(ph);
				ef->insertPoint(ph);
			}
		}

		// 3、根据得到的深度尺度归一化初始化两帧之间的平移参数，并将优化得到的位姿光度参数赋给初始化帧
		SE3 firstToNew = coarseInitializer->thisToNext;
		firstToNew.translation() /= rescaleFactor;

		// really no lock required, as we are initializing.
		{
			std::unique_lock<std::mutex> crlock(shellPoseMutex);
			coarseInitializer->firstFrame->shell->trackingRef = NULL;
			coarseInitializer->firstFrame->shell->camToTrackingRef = SE3();

			newFrame->shell->camToWorld = coarseInitializer->firstFrame->shell->camToWorld * firstToNew.inverse();
			newFrame->shell->aff_g2l = AffLight(0, 0);
			newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
			newFrame->shell->trackingRef = coarseInitializer->firstFrame->shell;
			newFrame->shell->camToTrackingRef = firstToNew.inverse();
		}

		initialized = true;
		printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)coarseInitializer->firstFrame->pointHessians.size());
	}

	/// <summary>
	/// 向系统中加入最新观测的图像进行定位建图
	/// step1：输入图像构造FrameHessian，并将shell数据加入到allFrameHistory中
	/// step2：将帧数据送入系统中，若未初始化则进行初始化，若已初始化则进行跟踪
	/// step3：跟踪成功后将被跟踪的帧数据发布到建图线程进行建图，并记录跟踪结果
	/// </summary>
	/// <param name="image">输入左目图像</param>
	/// <param name="image_right">输入右目图像</param>
	/// <param name="id">图像全局ID</param>
	void FullSystem::addActiveFrame(ImageAndExposure* image, ImageAndExposure* imageRight, int id)
	{
		printf("image id-stamp: %d-%.12f, marginalization total-half: %d-%d\n",
			id, pic_time_stamp[id], marg_num, marg_num_half);

		if (isLost) return;
		std::unique_lock<std::mutex> lock(trackMutex);

		if (use_stereo && (T_WD.scale() > 2 || T_WD.scale() < 0.6))
		{
			initFailed = true;
			first_track_flag = false;
			printf("tracking error scale\n");
		}

		if (!use_stereo && (T_WD.scale() < 0.1 || T_WD.scale() > 10))
		{
			initFailed = true;
			first_track_flag = false;
			printf("tracking error scale\n");
		}

		// 1、将输入图像构造Framehessian结构，并将帧shell数据加入allFrameHistory中
		FrameHessian* fh = new FrameHessian();
		FrameHessian* fhRight = new FrameHessian();
		FrameShell* shell = new FrameShell();

		shell->camToWorld = SE3();
		shell->aff_g2l = AffLight(0, 0);
		shell->marginalizedAt = shell->id = allFrameHistory.size();
		shell->timestamp = image->timestamp;
		shell->incomingId = id;
		fh->shell = shell;
		fhRight->shell = shell;

		fh->ab_exposure = image->exposure_time;
		fh->makeImages(image->image, &Hcalib);
		fhRight->ab_exposure = imageRight->exposure_time;
		fhRight->makeImages(imageRight->image, &Hcalib);
		fh->frameRight = fhRight;

		if (allFrameHistory.size() > 0)
		{
			fh->velocity = fh->shell->velocity = allFrameHistory.back()->velocity;
			fh->bias_g = fh->shell->bias_g = allFrameHistory.back()->bias_g + allFrameHistory.back()->delta_bias_g;
			fh->bias_a = fh->shell->bias_a = allFrameHistory.back()->bias_a + allFrameHistory.back()->delta_bias_a;
		}
		allFrameHistory.emplace_back(shell);

		// 2、将帧数据输入系统中，若系统未初始化则进行初始化，若初始化成功则进行跟踪
		// 双目初始化流程：提取第一帧图像的特征并在右目图像中跟踪，并初始化惯导信息完成初始化
		// 单目初始化流程：提取第一帧图像的特征并初始化惯导信息，在接下来输入系统的帧中找到位
		//				   移较大的帧并且接下来连续5帧都可以成功跟踪则完成初始化
		if (!initialized)
		{
			if (coarseInitializer->frameID < 0 && use_stereo)
			{
				coarseInitializer->setFirstStereo(&Hcalib, fh, fhRight);
				initFirstFrameImu(fh);
				initializeFromInitializer(fh);
				marg_num = 0;
				marg_num_half = 0;
			}
			else if (coarseInitializer->frameID < 0)
			{
				coarseInitializer->setFirst(&Hcalib, fh);
				initFirstFrameImu(fh);
			}
			else if (coarseInitializer->trackFrame(fh, outputWrapper))
			{
				initializeFromInitializer(fh);
				lock.unlock();							// 跟踪完毕，释放与跟踪线程绑定的互斥锁
				deliverTrackedFrame(fh, fhRight, true);	// 跟踪成功，将跟踪的帧提交进行建图线程进行建图操作
			}
			else
			{
				fh->shell->poseValid = false;
				delete fh;
				fh = NULL;
			}

			return;
		}
		else
		{
			// 系统中以滑窗中最后一帧作为coarseTraker的参考关键帧，建图线程会根据系统是否需要关键帧
			// 向滑窗中添加关键帧，此时最新关键帧发生变化，要求coarseTraker更新参考帧
			if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
			{
				std::unique_lock<std::mutex> crlock(coarseTrackerSwapMutex);
				CoarseTracker* tmp = coarseTracker;
				coarseTracker = coarseTracker_forNewKF;
				coarseTracker_forNewKF = tmp;
			}

			// coarseTracker handle跟踪最新进入系统的帧并根据返回值判断是否跟踪失败
			Vec4 tres = trackNewCoarse(fh);

			if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) ||
				!std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
			{
				printf("Initial Tracking failed: LOST!\n");
				isLost = true;
				return;
			}

			// 根据系统设定、帧位置变化情况、亮度变化情况以及帧间时间间隔，向系统中添加关键帧
			bool needToMakeKF = false;
			if (setting_keyframesPerSecond > 0)
			{
				needToMakeKF = allFrameHistory.size() == 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
			}
			else
			{
				Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

				double delta = 0;		// coarseTracker中参考帧与跟踪帧的像素位移
				double interval = 0;	// coarseTracker中参考帧与跟踪帧的时间间隔

				// 位置变化 & 亮度变化 & 时间间隔
				needToMakeKF = allFrameHistory.size() == 1 ||
					setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2 * coarseTracker->firstCoarseRMSE < tres[0];
				delta = setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
					setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0]));
				interval = pic_time_stamp[fh->shell->incomingId] - pic_time_stamp[coarseTracker->lastRef->shell->incomingId];

				if (interval >= 0.45 && delta > 0.5f) needToMakeKF = true;
			}

			// 3、跟踪成功后发布被跟踪的帧到建图线程中，发布跟踪结果到显示线程中并记录跟踪结果
			for (IOWrap::Output3DWrapper* ow : outputWrapper)
				ow->publishCamPose(fh->shell, &Hcalib);

			lock.unlock();
			deliverTrackedFrame(fh, fhRight, needToMakeKF);

			auto T = T_WD.matrix() * shell->camToWorld.matrix() * T_WD.inverse().matrix();
			savetrajectoryTum(SE3(T), run_time);

			return;
		}
	}

	/// <summary>
	/// 初始化惯导信息：包含设置第一帧坐标系为世界系以及定义世界系与DSO系之间的变换
	/// </summary>
	/// <param name="fh">进入系统的第一帧数据</param>
	void FullSystem::initFirstFrameImu(FrameHessian* fh)
	{
		int imuStartIdx = -1;
		int imuStartIdxGT = -1;
		double fhTime = pic_time_stamp[fh->shell->incomingId];

		imuStartIdx = findNearestIdx(imu_time_stamp, fhTime);
		imuStartIdxGT = findNearestIdx(gt_time_stamp, fhTime);

		if (imuStartIdx == -1) printf("timestamp error, check it\n");
		if (imuStartIdxGT == -1) printf("timestamp error, check it\n");
		if (imuStartIdx == -1 || imuStartIdxGT == -1) return;

		Vec3 g_b = Vec3::Zero();
		Vec3 g_w(0, 0, -1);

		for (int idx = 0; idx < 40; ++idx)
			g_b = g_b + m_acc[imuStartIdx - idx];

		g_b = -g_b / g_b.norm();
		Vec3 g_c = T_BC.inverse().rotationMatrix() * g_b;

		g_c = g_c / g_c.norm();
		Vec3 rAxis_wc = Sophus::SO3::hat(g_c) * g_w;

		// 罗德里格斯公式计算相机系到世界系的旋转
		double nNorm = rAxis_wc.norm();
		rAxis_wc = rAxis_wc / nNorm;
		double sin_theta = nNorm;
		double cos_theta = g_c.dot(g_w);

		Mat33 R_wc = sin_theta * Sophus::SO3::hat(rAxis_wc) +
			cos_theta * Mat33::Identity() + (1 - cos_theta) * rAxis_wc * rAxis_wc.transpose();

		// T_WR_align为GroundTruth与当前系统坐标系之间的变换
		// 这个参数的作用为将GroundTruth坐标转换到当前系统坐标系下
		SE3 T_wc(R_wc, Vec3::Zero());
		if (gt_path.empty()) T_WR_align = SE3();
		else T_WR_align = T_wc * T_BC.inverse() * gt_pose[imuStartIdxGT].inverse();

		fh->shell->camToWorld = T_wc;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);

		Mat33 R_wd = Mat33::Identity();
		T_WD = Sim3(RxSO3(1, R_wd), Vec3::Zero());
		T_WD_l = T_WD;
		T_WD_l_half = T_WD;
		state_twd.setZero();
	}

	/// <summary>
	/// 保存系统跟踪得到的位姿信息
	/// </summary>
	/// <param name="T">待保存的位姿信息</param>
	void FullSystem::savetrajectory(const Sophus::Matrix4d& T)
	{
		std::ofstream fs;
		std::string dsoPoseFile = "./data/" + savefile_tail + ".txt";
		fs.open(dsoPoseFile, std::ios::out | std::ios::app);

		fs << std::fixed << std::setprecision(9)
			<< T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " " << T(0, 3) << " "
			<< T(1, 0) << " " << T(1, 1) << " " << T(1, 2) << " " << T(1, 3) << " "
			<< T(2, 0) << " " << T(2, 1) << " " << T(2, 2) << " " << T(2, 3) << std::endl;
		fs.close();
	}

	/// <summary>
	/// 按照TUM格式保存系统跟踪得到的位姿信息
	/// </summary>
	/// <param name="T"></param>
	/// <param name="time"></param>
	void FullSystem::savetrajectoryTum(const SE3& T, const double time)
	{
		std::ofstream fs;
		std::string dsoPoseFile = "./data/" + savefile_tail + ".txt";
		fs.open(dsoPoseFile, std::ios::out | std::ios::app);

		Vec3 t = T.translation();
		Vec4 q = Eigen::Quaterniond(T.rotationMatrix()).coeffs();

		fs << std::fixed << std::setprecision(9) << time << " "
			<< t(0) << " " << t(1) << " " << t(2) << " "
			<< q(0) << " " << q(1) << " " << q(2) << " " << q(3) << std::endl;
		fs.close();

		int idxStart = -1;
		findNearestIdx(gt_time_stamp, time);
		assert(idxStart != -1);

		std::ofstream fsGT;
		std::string gtfile = "./data/" + savefile_tail + "_gt.txt";
		fsGT.open(gtfile, std::ios::out | std::ios::app);

		SE3 gt_C = T_WR_align * gt_pose[idxStart] * T_BC;
		t = gt_C.translation();
		q = Eigen::Quaterniond(gt_C.rotationMatrix()).coeffs();

		fsGT << std::fixed << std::setprecision(9) << gt_time_stamp[idxStart] << " "
			<< t(0) << " " << t(1) << " " << t(2) << " "
			<< q(0) << " " << q(1) << " " << q(2) << " " << q(3) << std::endl;
		fsGT.close();
	}

	/// <summary>
	/// 系统跟踪最新帧成功后，根据系统当前是否需要关键帧进行对应的建图操作；
	/// 若最新帧是关键帧，进行makeKeyFrame()的操作；
	/// 若最新帧不是关键帧，进行makeNonKeyFrame()的操作；
	/// </summary>
	/// <param name="fh">双目最新帧左目帧</param>
	/// <param name="fhRight">双目最新帧右目帧</param>
	/// <param name="needKF">系统是否需要关键帧</param>
	void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fhRight, bool needKF)
	{
		// 通过外部输入控制已跟踪的帧逐帧进行建图操作
		if (linearizeOperation)
		{
			if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
			{
				MinimalImageF3 img(wG[0], hG[0], fh->dI);
				IOWrap::displayImage("frameToTrack", &img);
				while (true)
				{
					char k = IOWrap::waitKey(0);
					if (k == ' ') break;
					handleKey(k);
				}
				lastRefStopID = coarseTracker->refFrameID;
			}
			else handleKey(IOWrap::waitKey(1));

			if (needKF) makeKeyFrame(fh, fhRight);
			else makeNonKeyFrame(fh, fhRight);
		}
		// 跟踪线程与建图线程间通过互斥锁自动进行建图操作
		else
		{
			std::unique_lock<std::mutex> lock(trackMapSyncMutex);
			unmappedTrackedFrames.emplace_back(fh);
			if (needKF) needNewKFAfter = fh->shell->trackingRef->id;
			trackedFrameSignal.notify_all();

			// 系统单目初始化的时候这两个handle的refFrameID同时为-1，这部分仅在初始化时会被调用
			while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
			{
				mappedFrameSignal.wait(lock);
			}

			lock.unlock();
		}
	}

	/// <summary>
	/// 建图线程Loop函数：每当跟踪线程跟踪成功提交最新帧后，
	/// 建图线程开始工程，并在工作完成后告知跟踪线程
	/// </summary>
	void FullSystem::mappingLoop()
	{
		std::unique_lock<std::mutex> lock(trackMapSyncMutex);

		while (runMapping)
		{
			while (unmappedTrackedFrames.empty())
			{
				trackedFrameSignal.wait(lock);
				if (!runMapping) return;
			}

			FrameHessian* fh = unmappedTrackedFrames.front();
			unmappedTrackedFrames.pop_front();
			FrameHessian* fhRight = unmappedTrackedFramesRight.front();
			unmappedTrackedFramesRight.pop_front();

			// 对于系统最开始跟踪的几帧都设置为关键帧
			if (allKeyFramesHistory.size() <= 2)
			{
				lock.unlock();
				makeKeyFrame(fh, fhRight);
				lock.lock();
				mappedFrameSignal.notify_all();
				continue;
			}

			if (unmappedTrackedFrames.size() > 3)
				needToKetchupMapping = true;

			if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
			{
				lock.unlock();
				makeNonKeyFrame(fh, fhRight);
				lock.lock();

				if (needToKetchupMapping && unmappedTrackedFrames.size() > 0)
				{
					FrameHessian* fh = unmappedTrackedFrames.front();
					unmappedTrackedFrames.pop_front();
					{
						std::unique_lock<std::mutex> crlock(shellPoseMutex);
						assert(fh->shell->trackingRef != 0);
						fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
						fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
					}
					delete fh;
				}
			}
			else
			{
				// 1、系统中已设定一直需要关键帧
				// 2、最新帧参考关键帧ID大于等于滑窗关键帧最后一帧ID
				if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
				{
					lock.unlock();
					makeKeyFrame(fh, fhRight);
					needToKetchupMapping = false;
					lock.lock();
				}
				else
				{
					lock.unlock();
					makeNonKeyFrame(fh, fhRight);
					lock.lock();
				}
			}
			mappedFrameSignal.notify_all();
		}
		printf("MAPPING FINISHED!\n");
	}

	/// <summary>
	/// 阻塞跟踪线程（主线程），等待建图线程完成任务
	/// </summary>
	void FullSystem::blockUntilMappingIsFinished()
	{
		std::unique_lock<std::mutex> lock(trackMapSyncMutex);
		runMapping = false;
		trackedFrameSignal.notify_all();
		lock.unlock();

		mappingThread.join();
	}

	/// <summary>
	/// 最新跟踪的帧fh不是关键帧时，设置该帧的位姿以及光度参数并且使用该帧追踪滑窗关键帧
	/// 中的未成熟点，优化未成熟点的精度
	/// </summary>
	/// <param name="fh"></param>
	/// <param name="fhRight"></param>
	void FullSystem::makeNonKeyFrame(FrameHessian* fh, FrameHessian* fhRight)
	{
		// 1、根据跟踪结果设置帧fh的位姿以及光度参数
		{
			std::unique_lock<std::mutex> crlock(shellPoseMutex);
			assert(fh->shell->trackingRef != 0);
			fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
			fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
		}

		// 2、滑窗中关键帧管理的未成熟点跟踪最新帧fh优化未成熟点的精度
		traceNewCoarse(fh);

		// 3、帧fh不能成为关键帧，那么在用完fh后需要将其内存空间释放掉
		delete fh;
		fh = NULL;
		delete fhRight;
		fhRight = NULL;
	}

	/// <summary>
	/// 最新跟踪的帧fh是关键帧时，系统进行滑窗优化并将边缘化的帧信息以及特征信息以舒尔补形式加入到优化中；
	/// step1：设置最新跟踪帧的线性化点处位姿以及光度参数并让所有未成熟点都跟踪一下最新帧提升特征精度；
	/// step2：标记需边缘化的滑窗关键帧并向优化函数中添加关键帧、特征以及残差，从而进行滑窗优化；
	/// step3：进行滑窗优化并根据返回的优化残差判断初始化结果的好坏，并去除被观测次数为零的特征；
	/// step4：对于标记为边缘化的帧和特征，计算其舒尔补保证信息不丢失，并在成员中去除帧和特征的结构；
	/// </summary>
	/// <param name="fh">最新跟踪的帧</param>
	/// <param name="fhRight">最新跟踪的帧的右目帧</param>
	void FullSystem::makeKeyFrame(FrameHessian* fh, FrameHessian* fhRight)
	{	
		// 1、设置最新帧的位姿以及光度参数，并在最新帧中跟踪滑窗关键帧管理的未成熟点，提升其精度
		{
			std::unique_lock<std::mutex> crlock(shellPoseMutex);
			assert(fh->shell->trackingRef != 0);
			fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
			fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
		}
		traceNewCoarse(fh);

		std::unique_lock<std::mutex> lock(mapMutex);

		// 2、根据设定的准则确定滑窗中需要边缘化的帧并将最新跟踪到的帧加入滑窗关键帧中
		flagFramesForMarginalization(fh);

		fh->idx = frameHessians.size();
		frameHessians.emplace_back(fh);
		fh->frameID = allKeyFramesHistory.size();
		fh->frameRight->frameID = 10000 + allKeyFramesHistory.size();
		allKeyFramesHistory.emplace_back(fh->shell);

		// 3、向EnergyFunctio中添加关键帧、添加残差并激活更多的特征，进行基于滑窗的优化操作
		ef->insertFrame(fh, &Hcalib);
		setPrecalcValues();

		// 最新关键帧进入滑窗后，原来滑窗中管理的特征在最新关键帧中也有观测值，为
		// 原来滑窗中管理的特征在最新关键帧与特征主帧间构造光度观测残差
		int numFwdResAdde = 0;
		for (FrameHessian* itFh : frameHessians)
		{
			if (itFh == fh) continue;
			for (PointHessian* ph : itFh->pointHessians)
			{
				PointFrameResidual* r = new PointFrameResidual(ph, itFh, fh);
				r->setState(ResState::INNER);
				ph->residuals.emplace_back(r);
				ef->insertResidual(r);
				ph->lastResiduals[1] = ph->lastResiduals[0];
				ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::INNER);
				numFwdResAdde += 1;
			}
		}
		printf("Add %d features in Sliding Window\n", numFwdResAdde);

		// 试着再激活一些特征，使得在滑窗优化中有充足的特征进行优化操作
		activatePointsMT();
		ef->makeIDX();

		// 滑窗优化的执行步骤，并返回优化残差
		fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
		float rmse = optimize(setting_maxOptIterations);

		// 系统的初始化结果至关重要，为了保证良好的初始化结果，在系统刚开始一段时间内，持续校验滑窗内的优化残差；
		// 若残差比较大说明系统初始化效果不好，应该放弃此次初始化；并且随着观测的关键帧数据增多，滑窗优化中用于
		// 优化的数据越多，此时滑窗优化的残差应该降低，以满足对初始化结果的有效校验
		if (allKeyFramesHistory.size() <= 4)
		{
			if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
			{
				printf("KeyFrames Size:%d,RMSE:%f,Initialize failed! Resetting.\n", allKeyFramesHistory.size(), rmse);
				initFailed = true;
			}
			if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
			{
				printf("KeyFrames Size:%d,RMSE:%f,Initialize failed! Resetting.\n", allKeyFramesHistory.size(), rmse);
				initFailed = true;
			}
			if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
			{
				printf("KeyFrames Size:%d,RMSE:%f,Initialize failed! Resetting.\n", allKeyFramesHistory.size(), rmse);
				initFailed = true;
			}
		}

		if (isLost) return;

		// 4、去除被观测次数为零的特征，并维护coarseTracker的参考关键帧始终为滑窗最后一帧
		removeOutliers();

		// 滑窗关键帧发生改变后，以最后一帧滑窗关键帧为跟踪基准的coarseTracker也要发生改变
		{
			std::unique_lock<std::mutex> crlock(coarseTrackerSwapMutex);
			coarseTracker_forNewKF->makeK(&Hcalib);
			coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fhRight, Hcalib);

			coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
			coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
		}

		debugPlot("post Optimize");

		// 5、滑窗优化完成后，为了维护滑窗中关键帧数量恒定，将需边缘化的帧和特征的信息以舒尔补形式加入Hessian和b
		flagPointsForRemoval();
		ef->dropPointsF();
		getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
			ef->lastNullspaces_affA, ef->lastNullspaces_affB);

		// 边缘化精度较高的特征，将信息保留在关键帧中
		ef->marginalizePointsF();

		// 在最新滑窗关键帧中提取更多的特征，避免特征随着跟踪进行越来越少
		makeNewTraces(fh, NULL);

		for (IOWrap::Output3DWrapper* ow : outputWrapper)
		{
			ow->publishGraph(ef->connectivityMap);
			ow->publishKeyframes(frameHessians, false, &Hcalib);
		}

		// 边缘化关键帧
		for (unsigned int it = 0; it < frameHessians.size(); ++it)
		{
			if (frameHessians[it]->flaggedForMarginalization)
			{
				marginalizeFrame(frameHessians[it]);
				it = 0;
			}
		}
	}

	/// <summary>
	/// 在最新关键帧中使用pixelSelector选取特征，避免特征越跟踪越少
	/// </summary>
	/// <param name="newFrame">最新关键帧</param>
	/// <param name="gtDepth">GroundTruth Depth</param>
	void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
	{
		pixelSelector->allowFast = true;
		int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

		newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
		newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
		newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

		for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; ++y)
		{
			for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; ++x)
			{
				int i = x + y * wG[0];
				if (selectionMap[i] == 0) continue;

				auto impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
				if (!std::isfinite(impt->energyTH)) delete impt;
				else newFrame->immaturePoints.emplace_back(impt);
			}
		}
	}

	/// <summary>
	/// 计算滑窗关键帧之间的相对关系
	/// </summary>
	void FullSystem::setPrecalcValues()
	{
		for (FrameHessian* fh : frameHessians)
		{
			fh->targetPrecalc.resize(frameHessians.size());
			for (unsigned int it = 0; it < frameHessians.size(); ++it)
				fh->targetPrecalc[it].set(fh, frameHessians[it], &Hcalib);
		}

		ef->setDeltaF(&Hcalib);
	}

	void FullSystem::printLogLine()
	{
		if (frameHessians.size() == 0) return;

		if (!setting_debugout_runquiet)
			printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
				allKeyFramesHistory.back()->id,
				statistics_lastFineTrackRMSE,
				ef->resInA,
				ef->resInL,
				ef->resInM,
				(int)statistics_numForceDroppedResFwd,
				(int)statistics_numForceDroppedResBwd,
				allKeyFramesHistory.back()->aff_g2l.a,
				allKeyFramesHistory.back()->aff_g2l.b,
				frameHessians.back()->shell->id - frameHessians.front()->shell->id,
				(int)frameHessians.size());


		if (!setting_logStuff) return;

		if (numsLog != 0)
		{
			(*numsLog) << allKeyFramesHistory.back()->id << " " <<
				statistics_lastFineTrackRMSE << " " <<
				(int)statistics_numCreatedPoints << " " <<
				(int)statistics_numActivatedPoints << " " <<
				(int)statistics_numDroppedPoints << " " <<
				(int)statistics_lastNumOptIts << " " <<
				ef->resInA << " " <<
				ef->resInL << " " <<
				ef->resInM << " " <<
				statistics_numMargResFwd << " " <<
				statistics_numMargResBwd << " " <<
				statistics_numForceDroppedResFwd << " " <<
				statistics_numForceDroppedResBwd << " " <<
				frameHessians.back()->aff_g2l().a << " " <<
				frameHessians.back()->aff_g2l().b << " " <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " " <<
				(int)frameHessians.size() << " " << "\n";
			numsLog->flush();
		}
	}

	void FullSystem::printEigenValLine()
	{
		if (!setting_logStuff) return;
		if (ef->lastHS.rows() < 12) return;

		MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
		MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
		int n = Hp.cols() / 8;
		assert(Hp.cols() % 8 == 0);

		// sub-select
		for (int i = 0; i < n; i++)
		{
			MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
			Hp.block(i * 6, 0, 6, n * 8) = tmp6;

			MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
			Ha.block(i * 2, 0, 2, n * 8) = tmp2;
		}
		for (int i = 0; i < n; i++)
		{
			MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
			Hp.block(0, i * 6, n * 8, 6) = tmp6;

			MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
			Ha.block(0, i * 2, n * 8, 2) = tmp2;
		}

		VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
		VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
		VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
		VecX diagonal = ef->lastHS.diagonal();

		std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
		std::sort(eigenP.data(), eigenP.data() + eigenP.size());
		std::sort(eigenA.data(), eigenA.data() + eigenA.size());

		int nz = std::max(100, setting_maxFrames * 10);

		if (eigenAllLog != 0)
		{
			VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
			(*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenAllLog->flush();
		}
		if (eigenALog != 0)
		{
			VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
			(*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenALog->flush();
		}
		if (eigenPLog != 0)
		{
			VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
			(*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenPLog->flush();
		}

		if (DiagonalLog != 0)
		{
			VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
			(*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			DiagonalLog->flush();
		}

		if (variancesLog != 0)
		{
			VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
			(*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			variancesLog->flush();
		}

		std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
		(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
		for (unsigned int i = 0; i < nsp.size(); i++)
			(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
		(*nullspacesLog) << "\n";
		nullspacesLog->flush();

	}

	void FullSystem::printFrameLifetimes()
	{
		if (!setting_logStuff) return;

		std::unique_lock<std::mutex> lock(trackMutex);

		std::ofstream* lg = new std::ofstream();
		lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
		lg->precision(15);

		for (FrameShell* s : allFrameHistory)
		{
			(*lg) << s->id
				<< " " << s->marginalizedAt
				<< " " << s->statistics_goodResOnThis
				<< " " << s->statistics_outlierResOnThis
				<< " " << s->movedByOpt;
			(*lg) << "\n";
		}

		lg->close();
		delete lg;
	}
}
