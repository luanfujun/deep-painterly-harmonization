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

if cores==1
  algo = 'cpu';
else
  algo = 'cputiled';
end

A=imread('in.png');
B=imread('tar.png');
A = imhistmatch(A, B);
figure; imshow(A)
figure; imshow(B)
B=B(1:end-1,1:end-1,:);

patch_w = 7;
ann0 = nnmex(A, B, algo, [], [], [], [], [], [], cores);   % Warm up

% Benchmark
tic;
nnmex(A, B, algo, [], [], [], [], [], [], cores);
nnmex(B, A, algo, [], [], [], [], [], [], cores);
disp(['NN A <-> B time: ', num2str(toc), ' sec']);

% Display field
ann = nnmex(A, B, algo, [], [], [], [], [], [], cores);
bnn = nnmex(B, A, algo, [], [], [], [], [], [], cores);
writeim(ann(:,:,1), 'test1.png');
writeim(sqrt(double(ann(1:end-patch_w,1:end-patch_w,3))), 'test2.png');

% Display reconstruction
writeim(votemex(B, ann), 'test3.png')       % Coherence
writeim(votemex(B, ann, bnn), 'test4.png')  % BDS

% Test 3x3 patches, and clipping
patch_w = 3;
ann = nnmex(A, B, algo, patch_w, [], [], [], [], [], cores);
bnn = nnmex(B, A, algo, patch_w, [], [], [], [], [], cores);
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
annp = nnmex(A, B, algo, [], 0, [], [], [], [], cores, [], [], ann0);
disp(['Initial guess mode run for 0 iterations: ', num2str(toc), ' sec']);
writeim(votemex(B, annp), 'test5.png')       % Coherence

ann0(1:200,1:200) = 0;
tic;
annp = nnmex(A, B, algo, [], [], [], [], [], [], cores, [], [], ann0);
disp(['Initial guess mode run for 5 iterations: ', num2str(toc), ' sec']);
writeim(votemex(B, annp), 'test6.png')       % Coherence


%close all;
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
