for i = 0 : 34
    close all;
    fn = [int2str(i) '_c_mask.png']
    I = im2double(imread(fn));
    [h w c] = size(I);
    if c == 3 
        I = I(:,:,1);
    end 
    
    h1 = h;
    w1 = w;
    if h ~= 700 && w ~= 700 
        if h > w 
            h1 = 700;
            w1 = floor(h1 * w/h);
        else 
            w1 = 700;
            h1 = floor(w1 * h/w);
        end 
    end 
    
    figure; imshow(I)
    r = 35;
    h = fspecial('gaussian', [r r], r/3);
    J = imfilter(I, h, 'same');
    
    figure; imshow(J)
    J2 = J;
    J2(J>0.1) = 1;
    J2(J<=0.1) = 0;
    figure; imshow(J2)
    imwrite(J2, [int2str(i) '_c_mask_dilated.png']);
end 