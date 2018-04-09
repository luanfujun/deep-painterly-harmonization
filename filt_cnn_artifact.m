% parpool(8)
% parfor i = 0 : 43
for i = 2:2 % 70:72
% parfor i = 0 : 149
    close all
    
     in_fn  = [int2str(i) '_result.jpg'];
     out_fn = [int2str(i) '_result2.jpg'];
     
 
%     in_fn  = ['../tmp36_wikiart2_current_more/' int2str(i) '_result.png'];
%     out_fn = ['../tmp36_wikiart2_current_more/' int2str(i) '_result2.png'];
    
   
    I = im2double(imread(in_fn));
    
     G = im2double(imread(['free2use/' int2str(i) '_naive.png']));
     M = im2double(imread(['free2use/' int2str(i) '_c_mask.png']));
     B = im2double(imread(['free2use/' int2str(i) '_target.jpg']));
 
    
    tr= 3;
    h = fspecial('gaussian', [2*tr+1 2*tr+1], tr);
    sM = imfilter(M, h, 'same'); 
    sM(sM > 0.01) = 1; % dialte
    sM(sM < 0.01) = 0;
    sM = imfilter(sM, h, 'same'); 

    addpath code/colorspace
    I_lab = colorspace('rgb->lab', I);

    addpath code/guided_filter
    addpath code/patchmatch-2.0

    r = 2; % try r=2, 4, or 8
    eps = 0.1^2; % try eps=0.1^2, 0.2^2, 0.4^2

    O_lab = I_lab;
    O_lab(:,:,2) = guidedfilter_color(G, I_lab(:,:,2), r, eps);
    O_lab(:,:,3) = guidedfilter_color(G, I_lab(:,:,3), r, eps);

    % runs here, deconvolution CNN artifact removed
    O1 = colorspace('lab->rgb', O_lab);
    % one patchmatch pass to further remove color artifact
    cores = 4; 
    algo = 'cputiled';
    patch_w = 7;
    ann = nnmex(O1, B, algo, patch_w, [], [], [], [], [], cores);
    O2_base = im2double(votemex(B, ann));
    
    r = 3;
    h = fspecial('gaussian', [2*r+1 2*r+1], r/3);
    O1_base = imfilter(O1, h, 'same');
    O2 = O2_base + O1 - O1_base;
    
    O2 = O2.*sM + B.*(1-sM);
    figure; imshow(I)
    figure; imshow(O2)

    imwrite(O2, out_fn);
end 


