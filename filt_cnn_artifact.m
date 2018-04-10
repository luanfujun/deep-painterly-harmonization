
for i = 0:34
    close all

    in_fn  = ['results/' int2str(i) '_final_res.png'];
    out_fn = ['results/' int2str(i) '_final_res2.png'];
    
    if exist(in_fn, 'file') ~= 2 
        fprintf('file doesn''t exist: %s\n', in_fn); 
        continue
    end 
    if exist(out_fn, 'file') == 2 
        fprintf('result already exists: %s\n', out_fn); 
        continue
    end 

    I = im2double(imread(in_fn));

    G = im2double(imread(['data/' int2str(i) '_naive.jpg']));
    M = im2double(imread(['data/' int2str(i) '_c_mask.jpg']));
    B = im2double(imread(['data/' int2str(i) '_target.jpg']));


    tr= 3;
    h = fspecial('gaussian', [2*tr+1 2*tr+1], tr);
    sM = imfilter(M, h, 'same'); 
    sM(sM > 0.01) = 1; % dialte
    sM(sM < 0.01) = 0;
    sM = imfilter(sM, h, 'same'); 

    addpath 3rdparty/colorspace
    I_lab = colorspace('rgb->lab', I);

    addpath 3rdparty/guided_filter
    addpath 3rdparty/patchmatch-2.0

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


