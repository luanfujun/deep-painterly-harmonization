
b=imread('b.png');
c=imread('c.png');
tic; cnn=nnmex(c,b,'rotscale',7,20); toc
imshow(cnn(:,:,1),[])
figure
imshow(cnn(:,:,4),[])

% Limiting to only unity scales on this input still works, as the object hasn't changed size too much
tic; cnn=nnmex(c,b,'rotscale',7,20, [], [], [], [], [], [], [], [], [], [], [], 1); toc
figure
imshow(cnn(:,:,1),[])
figure
imshow(cnn(:,:,4),[])
