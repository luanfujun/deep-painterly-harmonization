
b=imread('b.png');
c=imread('c.png');
tic; cnn=nnmex(c,b,'rotscale',7,20); toc
writeim(cnn(:,:,1),'test7.png')
writeim(cnn(:,:,4),'test8.png')

% Limiting to only unity scales on this input still works, as the object hasn't changed size too much
tic; cnn=nnmex(c,b,'rotscale',7,20, [], [], [], [], [], [], [], [], [], [], [], 1); toc
writeim(cnn(:,:,1),'test9.png')
writeim(cnn(:,:,4),'test10.png')
