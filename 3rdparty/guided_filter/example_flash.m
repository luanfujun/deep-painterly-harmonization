% example: flash/noflash denoising
% figure 8 in our paper
% *** Errata ***: there is a typo in the caption of figure 8, the eps should be 0.02^2 instead of 0.2^2; sig_r should be 0.02 instead of 0.2.

close all;

I = double(imread('.\img_flash\cave-flash.bmp')) / 255;
p = double(imread('.\img_flash\cave-noflash.bmp')) / 255;

r = 8;
eps = 0.02^2;

q = zeros(size(I));

q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);

figure();
imshow([I, p, q], [0, 1]);
