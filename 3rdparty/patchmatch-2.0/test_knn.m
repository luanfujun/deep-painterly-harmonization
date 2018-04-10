
cores = 2;

% kNN without enrichment
b=imread('b.png');
c=imread('c.png');
tic;
cnn=nnmex(b, b, 'cputiled', 7, 16, [], [], [], [], cores, [], [], [], [], [], 4);
toc
imshow(cnn(:,:,1,1),'test17.png', []); figure
imshow(cnn(:,:,1,2),'test18.png', []); figure
imshow(cnn(:,:,1,3),'test19.png', []); figure
imshow(cnn(:,:,1,4),'test20.png', []); figure
D = sqrt(double(cnn(:,:,3,:)));
format long;
disp(['Average dist (no enrichment):', num2str(mean(D(:)))]);

% kNN with enrichment -- both images must be the same. Enrichment requires the number of NN iterations to be even -- if not it will round down to the next even number.
tic;
cnn=nnmex(b, b, 'enrich', 7, 6, [], [], [], [], cores, [], [], [], [], [], 4);
toc
imshow(cnn(:,:,1,1),'test21.png', []); figure
imshow(cnn(:,:,1,2),'test22.png', []); figure
imshow(cnn(:,:,1,3),'test23.png', []); figure
imshow(cnn(:,:,1,4),'test24.png', [])
D = sqrt(double(cnn(:,:,3,:)));
disp(['Average dist (enrichment):', num2str(mean(D(:)))]);
