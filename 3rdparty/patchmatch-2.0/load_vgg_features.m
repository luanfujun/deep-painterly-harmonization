function [A_f, B_f] = load_vgg_features(exp_folder, id, A, B, layers, num_features)
    [hA, wA, cA] = size(A);
    [hB, wB, cB] = size(B);
    
    total_num_features = sum(num_features);
    A_f = zeros(hA, wA, total_num_features);
    B_f = zeros(hB, wB, total_num_features);
    
    index = 0;
    
    for i = 1 : length(layers)
        layer = layers{i}
        num_f = num_features(i)
           
        
        for j = 1 : num_f
            prefix = [exp_folder '\' layer '\content_' int2str(j)];
            content_f = load_img(prefix);
            prefix = [exp_folder '\' layer '\style_' int2str(j)];
            style_f = load_img(prefix);
            
            content_f = imresize(content_f, [hA, wA]);
            style_f   = imresize(style_f  , [hB, wB]);
            
            index = index + 1;
            A_f(:,:,index) = content_f;
            B_f(:,:,index) = style_f;
        end 
    end
    
end 

function img = load_img(prefix)
    img = im2double(imread([prefix '.png']));
    txt = load([prefix '.txt']);
    img = img * (txt(2) - txt(1)) + txt(1);
end 