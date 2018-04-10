%function ann = nnmex(A, B, [algo='cpu'], [patch_w=7], [nn_iters=5], [rs_max=100000], [rs_min=1], [rs_ratio=0.5], [rs_iters=1.0], [cores=2], [bmask=NULL]...
% [win_size=[INT_MAX INT_MAX]], [nnfield_prev=[]], [nnfield_prior=[]], [prior_winsize=[]], [knn=1], [scalerange=4])
%      (Pass [] to leave a parameter at its default).
%
%MATLAB Interface for PatchMatch + Generalized PatchMatch
%
%Input signals A, B, are 3D arrays of size either hxwx3 (image mode) or hxwxn (descriptor mode).
%  In image mode, the input images are broken up into overlapping patches of size patch_w x patch_w
%    (allowed data types: uint8, or floats in [0, 1], which are quantized).
%  In descriptor mode, the inputs have an n dimensional descriptor at each (x, y) coordinate
%    (allowed data types: uint8, or floats of any range, which are not quantized).
%    Pass a patch width of 1 unless you want these descriptors to be stacked in a larger patch_w^2 x n descriptor prior to distance comparison.
%
%Returns 'ann' - NN field (hxwx3, int32) mapping A -> B.
%Channel 1 is x coord, channel 2 is y coord, channel 3 is squared L2 distance.
% (In descriptor mode with float input, the output NN field reports all zeros in channel 3 for the output.)
% (When searching over all rotations+scales, the output NN field has channel 4 as theta 0...2pi, and channel 5 as scale. The patches in
%  image A are not rotated or scaled; the patches in image B are rotated and scaled around their center coordinate as if they had scale 1).
%
%The default distance metric to compare patches/descriptors is L2. To use L1 distance, set USE_L1 to 1 in patch.h.
%
%Options are:
%algo x         - One of 'cpu', 'gpucpu', 'cputiled', 'rotscale' (search over all rotations and scales), 'enrich' (an acceleration for kNN when both images are the same)
%patch_w p      - Width (and height) of patch, currently support sizes up to 32
%nn_iters n     - Iters of randomized NN algo
%rs_max w       - Maximum width for RS
%rs_min w       - Minimum width for RS
%rs_ratio r     - Ratio (< 1) of successive RS sizes
%rs_iters n     - Iters (double precision) of RS algorithm
%cores n        - Cores to run GPU-CPU algorithm on
%bmask [hxwx3]  - "1" indicates a hole
%win_size [w h] - Size of search window [2*h+1 x 2*w+1] around the input pixel location 
%				(interpolated linearly to the output coordinates in case of different
%				sizes). Slower but allows to limit the search space locally.
%ann_prev       - (hxwx3, double) initial mapping A -> B. The final result is
%				the minimum distance between initial mapping and random initialization + a
%				few final iterations. The squared distance channel in ann_prev is not used.
%ann_prior      - (hxwx2) field that constrains the search in a local window around the locations in B defined by the ann_prior field.
%ann_winsize    - (hxwx2) array matching ann_prior that defines locally the window size (per pixel) - first channel for window width, second channel for the height.
%knn            - Defines number of k-Nearest Neighbors to return. Returns a NNF of size h x w x 3 x k.
%scalerange     - When searching over rotations+scales, patches in image B can have size in [1/scalerange, scalerange].
%
%------------------------------------------------------------------------%
% Copyright 2008-2010 Adobe Systems Inc. and Connelly Barnes
%
% For noncommercial use only.
%
% Please cite the appropriate paper(s) if used in research:
%
% - PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing.
%   Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman.
%   ACM Transactions on Graphics (Proc. SIGGRAPH), 28(3), 2009
%   http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/
%
% - The Generalized PatchMatch Correspondence Algorithm
%   Connelly Barnes, Eli Shechtman, Dan B Goldman, and Adam Finkelstein.
%   Proc. European Conference on Computer Vision 2010
%   http://www.cs.princeton.edu/gfx/pubs/Barnes_2010_TGP/index.php
%   (k-Nearest Neighbors, Rotations+Scales, Descriptor Matching)
%
% Main contact: csbarnes@cs.princeton.edu  (Connelly)
% Version: 2.0, 2010-11-05
%------------------------------------------------------------------------%
