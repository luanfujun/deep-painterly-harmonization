% example: guided feathering
% figure 9 in our paper

close all;

I = double(imread('.\img_feathering\toy.bmp')) / 255;
p = double(rgb2gray(imread('.\img_feathering\toy-mask.bmp'))) / 255;

r = 60;
eps = 10^-6;

q = guidedfilter_color(I, p, r, eps);

figure();
imshow([I, repmat(p, [1, 1, 3]), repmat(q, [1, 1, 3])], [0, 1]);
