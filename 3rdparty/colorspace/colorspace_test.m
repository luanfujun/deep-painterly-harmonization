% Transform accuracy test
% Test script for colorspace

% Pascal Getreuer 2006-2010


fprintf(['\nTransform accuracy test\n\n',...
      'To verify the invertibility of the color transfomations, this test\n',...
      'transforms sRGB data to a space, inverts, and compares with the\n',...
      'original data.\n']);
N = 1e5;            % Number of points to test
A = rand(N,3);      % Generate points uniformly in the sRGB colorspace

% Include pure black and pure white
A(1,:) = 0;
A(2,:) = 1;

Space = {'YPbPr', 'YCbCr', 'JPEG-YCbCr', 'YDbDr', 'YIQ','YUV', 'HSV', ...
      'HSL', 'HSI', 'XYZ', 'Lab', 'Luv', 'LCH', 'CAT02 LMS'};
fprintf('\n Transform          RMSE Error   Max Error\n\n');

for k = 1:length(Space)
   B = colorspace([Space{k},'<-RGB'],A);  % Convert to Space{k}
   R = colorspace(['RGB<-',Space{k}],B);  % Convert back to sRGB
   RMSE = sqrt(mean((A(:) - R(:)).^2));
   MaxError = max(abs(A(:) - R(:)));
   fprintf(' RGB<->%-10s   %9.2e    %9.2e\n', Space{k}, RMSE, MaxError);
end

fprintf('\n\n');
