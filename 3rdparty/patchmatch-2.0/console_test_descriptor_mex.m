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

% Test 'image mode' nnmex, and votemex (see test_square.m and test_daisy.m for tests of 'descriptor mode' nnmex).

cores = 4;    % Use more cores for more speed

disp('start testing descriptor mode');

if cores==1
  algo = 'cpu';
else
  algo = 'cputiled';
end


A=imread('b.png');
B=imread('c.jpg');
B=B(1:end-1,1:end-1,:);

SA = A; SA(:,:,4:6) = A; SA(:,:,7:9) = A; SA = double(SA);
SB = B; SB(:,:,4:6) = B; SB(:,:,7:9) = B; SB = double(SB);
disp('data generated');

patch_w = 7;

% Display field
ann = nnmex(SA, SB, algo, [], [], [], [], [], [], cores);
bnn = nnmex(SB, SA, algo, [], [], [], [], [], [], cores);

% Display reconstruction
writeim(votemex(B, ann), 'test13.png')       % Coherence
writeim(votemex(B, ann, bnn), 'test14.png')  % BDS

