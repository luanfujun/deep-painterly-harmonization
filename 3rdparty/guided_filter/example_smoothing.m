% example: edge-preserving smoothing
% figure 1 in our paper

close all;

I = double(imread('.\img_smoothing\cat.bmp')) / 255;
p = I;
r = 4; % try r=2, 4, or 8
eps = 0.2^2; % try eps=0.1^2, 0.2^2, 0.4^2

q = guidedfilter(I, p, r, eps);

figure();
imshow([I, q], [0, 1]);
