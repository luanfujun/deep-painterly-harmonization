function out = writeim(A, filename)

a = min(A(:));
b = max(A(:));
A = uint8((A-a)*(255/(b-a)));
imwrite(A, filename);
