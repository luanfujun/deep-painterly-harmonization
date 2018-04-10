

N = 24;
patch = 7;

for i = 1 : N
    i
    exp_folder = ['D:\Dropbox\Public\StyleTransfer\GainMapResults\results_9_9\segs'];
    A = imread([exp_folder '\in' int2str(i) '.png']);
    B = imread([exp_folder '\tar' int2str(i) '.png']);
    
    layers = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'};
    num_features = [64, 128, 256, 512, 512];
    
    [A_f, B_f] = load_vgg_features([exp_folder '\segmentations_' int2str(i)], i, A, B, layers, num_features);
    
    [C, field] = neural_patchmatch(A, B, A_f, B_f, patch);
    
    imwrite(C, [exp_folder '\match' int2str(i) '.png']);
end 