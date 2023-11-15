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

#include <string>
#include <cmath>
#include <vector>
#include "NumType.h"

#ifdef WIN32
#include <sys/timeb.h>

struct timedso
{
	long	tv_sec;         /* seconds */
	long    tv_usec;        /* and microseconds */
};

inline int gettimeofday(timedso *tp, struct timezone *tzp)
{
	struct timeb timebuffer;
	ftime(&timebuffer);
	tp->tv_sec = timebuffer.time;
	tp->tv_usec = timebuffer.millitm * 1000;
	return 0;
}
#else
#include <sys/time.h>
#endif

namespace dso
{
#define SOLVER_SVD (int)1
#define SOLVER_ORTHOGONALIZE_SYSTEM (int)2
#define SOLVER_ORTHOGONALIZE_POINTMARG (int)4
#define SOLVER_ORTHOGONALIZE_FULL (int)8
#define SOLVER_SVD_CUT7 (int)16
#define SOLVER_REMOVE_POSEPRIOR (int)32
#define SOLVER_USE_GN (int)64
#define SOLVER_FIX_LAMBDA (int)128
#define SOLVER_ORTHOGONALIZE_X (int)256
#define SOLVER_MOMENTUM (int)512
#define SOLVER_STEPMOMENTUM (int)1024
#define SOLVER_ORTHOGONALIZE_X_LATER (int)2048
#define PYR_LEVELS 6

#define patternNum 8
#define patternP staticPattern[8]
#define patternPadding 2

#define SAVE_INITIALIZER_DATA

	// ============== PARAMETERS TO BE DECIDED ON COMPILE TIME =================
	extern int pyrLevelsUsed;

	extern float setting_keyframesPerSecond;
	extern bool setting_realTimeMaxKF;
	extern float setting_maxShiftWeightT;
	extern float setting_maxShiftWeightR;
	extern float setting_maxShiftWeightRT;
	extern float setting_maxAffineWeight;
	extern float setting_kfGlobalWeight;

	extern float setting_idepthFixPrior;
	extern float setting_idepthFixPriorMargFac;
	extern float setting_initialRotPrior;
	extern float setting_initialTransPrior;
	extern float setting_initialAffBPrior;
	extern float setting_initialAffAPrior;
	extern float setting_initialCalibHessian;
	extern float setting_initialIMUHessian;
	extern float setting_initialScaleHessian;
	extern float setting_initialbaHessian;
	extern float setting_initialbgHessian;

	extern int setting_solverMode;
	extern double setting_solverModeDelta;

	extern float setting_minIdepthH_act;
	extern float setting_minIdepthH_marg;

	extern int setting_initStepFrames;
	extern bool setting_useInitStep;
	extern bool setting_initFixAffine;

	extern float setting_maxIdepth;
	extern float setting_maxPixSearch;
	extern float setting_desiredImmatureDensity;
	extern float setting_desiredPointDensity;
	extern float setting_minPointsRemaining;
	extern float setting_maxLogAffFacInWindow;
	extern int setting_minFrames;
	extern int setting_maxFrames;
	extern int setting_minFrameAge;
	extern int setting_maxOptIterations;
	extern int setting_minOptIterations;
	extern float setting_thOptIterations;
	extern float setting_outlierTH;
	extern float setting_outlierTHSumComponent;

	extern int setting_pattern;
	extern float setting_margWeightFac;
	extern int setting_GNItsOnPointActivation;

	extern float setting_minTraceQuality;
	extern int setting_minTraceTestRadius;
	extern float setting_reTrackThreshold;

	extern int   setting_minGoodActiveResForMarg;
	extern int   setting_minGoodResForMarg;
	extern int   setting_minInlierVotesForMarg;

	extern int setting_photometricCalibration;
	extern bool setting_useExposure;
	extern float setting_affineOptModeA;
	extern float setting_affineOptModeB;
	extern int setting_gammaWeightsPixelSelect;

	extern bool setting_forceAceptStep;

	extern float setting_huberTH;

	extern bool setting_logStuff;
	extern float benchmarkSetting_fxfyfac;
	extern int benchmarkSetting_width;
	extern int benchmarkSetting_height;
	extern float benchmark_varNoise;
	extern float benchmark_varBlurNoise;
	extern int benchmark_noiseGridsize;
	extern float benchmark_initializerSlackFactor;

	extern float setting_frameEnergyTHConstWeight;
	extern float setting_frameEnergyTHN;

	extern float setting_frameEnergyTHFacMedian;
	extern float setting_overallEnergyTHWeight;
	extern float setting_coarseCutoffTH;

	extern float setting_minGradHistCut;
	extern float setting_minGradHistAdd;
	extern float setting_gradDownweightPerLevel;
	extern bool  setting_selectDirectionDistribution;

	extern float setting_trace_stepsize;
	extern int setting_trace_GNIterations;
	extern float setting_trace_GNThreshold;
	extern float setting_trace_extraSlackOnTH;
	extern float setting_trace_slackInterval;
	extern float setting_trace_minImprovementFactor;

	extern bool setting_render_displayCoarseTrackingFull;
	extern bool setting_render_renderWindowFrames;
	extern bool setting_render_plotTrackingFull;
	extern bool setting_render_display3D;
	extern bool setting_render_displayResidual;
	extern bool setting_render_displayVideo;
	extern bool setting_render_displayDepth;
	extern bool setting_fullResetRequested;
	extern bool setting_debugout_runquiet;

	extern bool setting_disableAllDisplay;
	extern bool setting_onlyLogKFPoses;
	extern bool setting_debugSaveImages;
	extern bool setting_multiThreading;

	extern int sparsityFactor;
	extern bool setting_goStepByStep;
	extern bool plotStereoImages;

	extern float freeDebugParam1;
	extern float freeDebugParam2;
	extern float freeDebugParam3;
	extern float freeDebugParam4;
	extern float freeDebugParam5;

	template <typename T>
	void freePointer(T* ptr)
	{
		if (ptr != NULL)
		{
			delete ptr;
			ptr = NULL;
		}
	}

	template <typename T>
	void freePointerVec(T* ptr)
	{
		if (ptr != NULL)
		{
			delete[] ptr;
			ptr = NULL;
		}
	}

	void handleKey(char k);
	int findNearestIdx(const std::vector <double>&, const double);

	extern int staticPattern[10][40][2];
	extern int staticPatternNum[10];
	extern int staticPatternPadding[10];
	
	extern std::string input_gtPath;
	extern std::string input_imuPath;
	extern std::string input_vignette;
	extern std::string input_gammaCalib;
	extern std::string input_sourceLeft;
	extern std::string input_sourceRight;
	extern std::string input_calibLeft;
	extern std::string input_calibRight;
	extern std::string input_calibStereo;
	extern std::string input_calibImu;
	extern std::string input_picTimestampLeft;
	extern std::string input_picTimestampRight;

	extern double setting_baseline;
	extern double setting_stereoWeight;
	extern double setting_imuWeightNoise;
	extern double setting_imuWeightTracker;
	extern bool setting_useStereo;
	extern double setting_margWeightFacImu;

	extern std::vector<SE3> input_gtPose;
	extern std::vector<Vec3> input_gtVelocity;
	extern std::vector<Vec3> input_gtBiasG;
	extern std::vector<Vec3> input_gtBiasA;
	extern std::vector<Vec3> input_gryList;
	extern std::vector<Vec3> input_accList;

	extern SE3 T_C0C1;
	extern SE3 T_C1C0;
	extern Mat33f K_right;
	extern std::vector<double> gt_time_stamp;
	extern std::vector<double> imu_time_stamp;
	extern std::vector<double> pic_time_stamp;
	extern std::vector<double> pic_time_stamp_r;
	extern SE3 T_BC;
	extern Mat33 GyrCov;
	extern Mat33 AccCov;
	extern Mat33 GyrRandomWalkNoise;
	extern Mat33 AccRandomWalkNoise;

	extern Sim3 T_WD;
	extern Sim3 T_WD_l;
	extern Sim3 T_WD_l_half;
	extern Sim3 T_WD_change;
	
	extern int index_align;
	extern SE3 T_WR_align;
	extern double run_time;
	extern Vec7 step_twd;
	extern Vec7 state_twd;
	
	extern bool setting_useImu;
	extern bool setting_imuTrackFlag;
	extern bool setting_imuTrackReady;
	extern bool setting_useOptimize;
	extern bool setting_useDynamicMargin;
	extern double setting_gravityNorm;
	extern double setting_dynamicMin;
	
	extern int marg_num;
	extern int marg_num_half;
	extern bool first_track_flag;
}