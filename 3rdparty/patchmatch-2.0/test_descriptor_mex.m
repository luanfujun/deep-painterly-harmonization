9i8u%------------------------------------------------------------------------%
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

disp('start testing descriptor mode');

if cores==1
  algo = 'cpu';
else
  algo = 'cputiled';
end


A=imread('a.png');
B=imread('b.png');
B=B(1:end-1,1:end-1,:);

SA = A; SA(:,:,4:6) = A; SA(:,:,7:9) = A; SA = double(SA);
SB = B; SB(:,:,4:6) = B; SB(:,:,7:9) = B; SB = double(SB);
disp('data generated');

patch_w = 7;
ann0 = nnmex(SA, SB, algo, [], [], [], [], [], [], cores);   % Warm up
disp('warm-up finished');

% Benchmark
tic;
nnmex(SA, SB, algo, [], [], [], [], [], [], cores);
nnmex(SB, SA, algo, [], [], [], [], [], [], cores);
disp(['NN A <-> B time: ', num2str(toc), ' sec']);

% Display field
ann = nnmex(SA, SB, algo, [], [], [], [], [], [], cores);
bnn = nnmex(SB, SA, algo, [], [], [], [], [], [], cores);
imshow(ann(:,:,1), []);
figure
imshow(sqrt(abs(double(ann(1:end-patch_w,1:end-patch_w,3)))), []);
figure

% Display reconstruction
imshow(votemex(B, ann))       % Coherence
figure
imshow(votemex(B, ann, bnn))  % BDS

% Test 3x3 patches, and clipping
patch_w = 3;
ann = nnmex(SA, SB, algo, patch_w, [], [], [], [], [], cores);
bnn = nnmex(SB, SA, algo, patch_w, [], [], [], [], [], cores);
% figure
ann(1,1,1) = -2000;
ann(2,1,1) = 2000;
ann(3,1,2) = 2000;
ann(4,1,2) = -2000;
bnn(1,1,1) = -2000;
bnn(2,1,1) = 2000;
bnn(3,1,2) = 2000;
bnn(4,1,2) = -2000;
% imshow(votemex(B, ann, bnn, algo, patch_w));

% Test initial guess
tic;
annp = nnmex(SA, SB, algo, [], 0, [], [], [], [], cores, [], [], ann0);
disp(['Initial guess mode run for 0 iterations: ', num2str(toc), ' sec']);
figure
imshow(votemex(B, annp))       % Coherence

ann0(1:200,1:200) = 0;
tic;
annp = nnmex(SA, SB, algo, [], [], [], [], [], [], cores, [], [], ann0);
disp(['Initial guess mode run for 5 iterations: ', num2str(toc), ' sec']);
figure
imshow(votemex(B, annp))       % Coherence


close all;
%% Test for memory leaks
%A=imresize(A,0.25);
%B=imresize(B,0.25);
%B=B(1:end-1,1:end-1,:);
%ann = nnmex(A, B); bnn = nnmex(B, A);
%
%user = memory;
%disp(['before memory leak test: ', num2str(user.MemUsedMATLAB/1e6), ' MB']);
%for i=1:100
%  ann = nnmex(A, B); bnn = nnmex(B, A);
%end
%user = memory;
%disp(['after memory leak test: ', num2str(user.MemUsedMATLAB/1e6), ' MB']);
