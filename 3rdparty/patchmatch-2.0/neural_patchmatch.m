function [C, field] = neural_patchmatch(A, B, A_f, B_f, patch_w)
    cores = 2;
    
    if cores==1
        algo = 'cpu';
    else
        algo = 'cputiled';
    end
    
    field = nnmex(A_f, B_f, algo, patch_w, [], [], [], [], [], cores);
    
    C = votemex(B, field);
    writeim(C, 'C.png')
    
    disp('done!')
end 