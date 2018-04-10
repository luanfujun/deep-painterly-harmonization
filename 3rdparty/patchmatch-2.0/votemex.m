%function A = votemex(B, ann, [bnn=[]], [algo='cpu'], [patch_w=7],
%[bmask=[]], [bweight=[]], [coherence_weight=1], [complete_weight=1],
%[amask=[]], [aweight=[]]), [A0=[]]
%
%
%Input image B is hxwx3, floats in [0, 1] or uint8.
%Input fields are:
% ann = nnmex(A, B)
% bnn = nnmex(B, A)
%If bnn unspecified, uses coherence only.
%
%Options are:
%algo             - One of 'cpu', 'cputiled'
%patch_w          - Width (and height) of patch, currently support sizes up to 32
%bmask            - B mask image, votes made only where mask is zero
%                   (affects both 'coherence' and 'completeness'
%bweight          - B weight image, weights are 32 bit floats, and
%                   correspond to the *center* of the patch
%coherence_weight - Weight multiplier for coherence
%complete_weight  - Weight multiplier for completeness
%amask            - A mask image, votes made only where mask is zero
%                   (affects only 'coherence')
%aweight          - A weight image, if not given uses inverse mapping
%                   and B weight image. Corresponds to the *center* of the
%                   patch.
%A0               - Initial guess for A. It is used as default in regions
%                   with no votes - outside the output mask 'amask' and
%                   with no votes from the input.
%
%NOTE: all NN fields and masks correspond to the upper-left corner of the
%patches, and the field indeces are in C coordinates (start with [0,0]),
%wheras the weight maps correspond to the *center* of the patch.
%
%
%------------------------------------------------------------------------%
% Copyright 2008-2009 Adobe Systems Inc., for noncommercial use only.
% Citation:
%   Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman.
%   PatchMatch: A Randomized Correspondence Algorithm for Structural Image
%   Editing. ACM Transactions on Graphics (Proc. SIGGRAPH), 28(3), 2009
%   http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/
% Main contact: csbarnes@cs.princeton.edu  (Connelly)
% Version: 1.0, 21-June-2008
%------------------------------------------------------------------------%

