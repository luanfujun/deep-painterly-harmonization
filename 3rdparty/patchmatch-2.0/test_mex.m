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

cores = 2;    % Use more cores for more speed

if cores==1
  algo = 'cpu';
else
  algo = 'cputiled';
end

A=imread('a.jpg');
B=imread('b.jpg');
B=B(1:end-1,1:end-1,:);

patch_w = 3;
ann0 = nnmex(A, B, algo, [], [], [], [], [], [], cores);   % Warm up

% Benchmark
tic;
nnmex(A, B, algo, [], [], [], [], [], [], cores);
nnmex(B, A, algo, [], [], [], [], [], [], cores);
disp(['NN A <-> B time: ', num2str(toc), ' sec']);

% Display field
ann = nnmex(A, B, algo, [], [], [], [], [], [], cores);
bnn = nnmex(B, A, algo, [], [], [], [], [], [], cores);
imshow(ann(:,:,1), []);
figure
imshow(sqrt(double(ann(1:end-patch_w,1:end-patch_w,3))), []);
figure

% Display reconstruction
imshow(votemex(B, ann))       % Coherence
figure
imshow(votemex(B, ann, bnn))  % BDS
