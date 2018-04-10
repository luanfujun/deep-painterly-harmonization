
--------------------------------------------------------------------------------------------------

PatchMatch -- Core matching algorithm only
MATLAB Mex Version 2.0 (2010-11-05)

  By Connelly Barnes
  Copyright 2008-2010 Adobe Systems Inc and Connelly Barnes
  Licensed by Adobe for noncommercial research use only.

--------------------------------------------------------------------------------------------------
Background
--------------------------------------------------------------------------------------------------

This code implements a fast randomized matching algorithm described in two publications as part of
my Ph.D. thesis at Princeton.

The algorithm solves the following problem: For each patch (a small fixed size rectangular region,
e.g. 7x7) in image A, find the nearest patch in image B, where "nearest" is defined as say the L2
distance between corresponding RGB tuples. Our algorithm converges quickly to an approximate
solution. In our original publication we find only 1 nearest neighbor, for patches that translate
only. In our later work (Generalized PatchMatch) we extend to k-Nearest Neighbors (k-NN), and allow
patches to rotate+scale, and match arbitrary descriptors (tuples of an arbitrary dimension rather
than just RGB, e.g. one can match densely computed SIFT descriptors).

For more information on the problem we are solving and our solution see:

 - PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing
   SIGGRAPH 2009, Connelly Barnes, Eli Shechtman, Adam Finkelstein, Dan B Goldman
   http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php

 - The Generalized PatchMatch Correspondence Algorithm
   ECCV 2010, Connelly Barnes, Eli Shechtman, Dan B Goldman, Adam Finkelstein
   http://www.cs.princeton.edu/gfx/pubs/Barnes_2010_TGP/index.php

Please cite these paper(s) if you use this code in an academic publication.

--------------------------------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------------------------------

 - For a portable, unoptimized, but very easy to understand PatchMatch code (200 lines, only 100
   in the core algorithm), see: pm_minimal.cpp. This can be built without any dependencies, and
   runs only requiring that ImageMagick be installed on your Windows/Mac/Unix machine.

 - For an optimized and more powerful PatchMatch, build the MATLAB interface:

   Build: Use build_windows.bat / build_mac.sh / build_unix.sh for your system
   (Known to work on Windows XP, Vista, Mac OS X, and Linux, after varying amounts of fighting with the compiler).
   (Disable optimizations if you're just trying to get it to build, this will help speed up your build process).
   (OpenMP is used, but not required -- set USE_OPENMP in nn.h to 0 if your compiler does not support OpenMP)

   Usage: do 'help nnmex' or 'help votemex' in MATLAB, or run test_mex.m, test_rot_scale.m, test_descriptor_mex.m, or test_knn.m.

   For efficiency, use the 'cputiled' algorithm and set the cores argument to the number of processor cores on your system.

--------------------------------------------------------------------------------------------------
Version History
--------------------------------------------------------------------------------------------------

 * Wish list (to be added soon):
   - Window constraints for kNN matching, or minimal distance between the kNN matches, to keep matches from being in nearby spatial locations.
   - Examples: Object detection
   - C commandline interface (not requiring MATLAB)
   - Further optimizations (SSE, precomputed random search samples)

 * Version 2.0 (2010-11-05)
   Connelly Barnes
   - Added features from "The Generalized PatchMatch Correspondence Algorithm"
   - Exposed k-Nearest Neighbor matching, matching across rotations+scales, and enrichment (a feature which accelerates convergence of kNN)
   - Fixed multicore tiled algorithm to not have contention issues.
   - Exposed min and max scale and rotation as parameters for rotation+scale matching.
   - Minimal 200 line implementation of PatchMatch, with no build dependencies, provided in pm_minimal.cpp.

 * Version 1.1 (2010-06-30)
   Xiaobai Chen
   - New solution file "mex.sln" with two projects "nnmex" and "votemex" (important to set /openmp, otherwise must disable those omp_* calls)
   - nnmex with bmask (changed the interface of nnmex)
   - "clip_value" function to replace range examination code in mexutil.cpp and mexutil.h
   - Added specialized templates for larger patches (up to 32)
   - Need to guarantee patch size is less than the image size (no guarantee in the C++ code)
   - New descriptor mode to support patches
   - Re-implemented functions start with "XC", places with minor changes are not always marked
   - Updated nnmex.m and votemex.m

 * Version 1.0 (2009-10-23)
   Connelly Barnes Initial Release
