% Demo for colorspace.m - the CIE xyY "tongue"

% Pascal Getreuer 2006

N = 800;
Nx = round(N*0.736);
Ny = round(N*0.84);
% Generate colors in the xyY color space
x = linspace(0,0.736,Nx);
y = linspace(0,0.84,Ny);
[xx,yy] = meshgrid(x,y);
% i = (xx + yy <= 1.01 & (xx - 0.2)/2 - yy <= 0.02);

% Convert from xyY to XYZ
a = 0.6491;
X = a*xx;
Y = a*yy;
Z = a*(1 - xx - yy);

Corner = colorspace('xyz<-rgb', [1,0,0;0,1,0;0,0,1]);
Corner = Corner(:,1:2)./repmat(sum(Corner,2),1,2);

v0x = Corner(3,1) - Corner(1,1);
v0y = Corner(3,2) - Corner(1,2);
v1x = Corner(2,1) - Corner(1,1);
v1y = Corner(2,2) - Corner(1,2);
v2x = xx - Corner(1,1);
v2y = yy - Corner(1,2);

dot00 = v0x*v0x + v0y*v0y;
dot01 = v0x*v1x + v0y*v1y;
dot02 = v0x*v2x + v0y*v2y;
dot11 = v1x*v1x + v1y*v1y;
dot12 = v1x*v2x + v1y*v2y;
Denom = dot00*dot11 - dot01*dot01;
bu = (dot11*dot02 - dot01*dot12) / Denom;
bv = (dot00*dot12 - dot01*dot02) / Denom;


i = (bu > 0 & bv > 0 & bu + bv < 1);
ie = conv2(double(i),[0,1,0;1,1,1;0,1,0],'same');

a = 0.3 + ie*0.7/5;
X = a.*X;
Y = a.*Y;
Z = a.*Z;

% Convert from XYZ to R'G'B'
Color = colorspace('rgb<-xyz',cat(3,X,Y,Z));
Color = min(max(Color,0),1);

% Render the colors on the tongue
clf;
plot(xx,yy,'b.');
image(x,y,Color)
colormap(gray);
set(gca,'YDir','normal');
axis image

% The boundary of the tongue
xy = [[0.1740, 0.0050];[0.1736, 0.0049];[0.1703, 0.0058];
   [0.1566, 0.0177];[0.1440, 0.0297];[0.1241, 0.0578];
   [0.1096, 0.0868];[0.0913, 0.1327];[0.0687, 0.2007];
   [0.0454, 0.2950];[0.0235, 0.4127];[0.0082, 0.5384];
   [0.0039, 0.6548];[0.0139, 0.7502];[0.0389, 0.8120];
   [0.0743, 0.8338];[0.1142, 0.8262];[0.1547, 0.8059];
   [0.1929, 0.7816];[0.2296, 0.7543];[0.2658, 0.7243];
   [0.3016, 0.6923];[0.3373, 0.6589];[0.3731, 0.6245];
   [0.4087, 0.5896];[0.4441, 0.5547];[0.4788, 0.5202];
   [0.5125, 0.4866];[0.5448, 0.4544];[0.5752, 0.4242];
   [0.6029, 0.3965];[0.6270, 0.3725];[0.6482, 0.3514];
   [0.6658, 0.3340];[0.6801, 0.3197];[0.7006, 0.2993];
   [0.7140, 0.2859];[0.7260, 0.2740];[0.7340, 0.2660]];
% Make a smooth boundary with spline interpolation
xi = [interp1(xy(:,1),1:0.25:size(xy,1),'spline'),xy(1,1)];
yi = [interp1(xy(:,2),1:0.25:size(xy,1),'spline'),xy(1,2)];

% Draw the boundary of the tongue
hold on;
% set(patch([0.8;Corner([1,2,3,1],1);0.8;0.8;0;0;0.8],...
%     [0.9;Corner([1,2,3,1],2);0.9;0;0;0.9;0.9],...
%    [0,0,0] + 0.3),'EdgeColor','none');
set(patch([0.8,-1e2,-1e2,0.8,xi(:).',0.8],[0.9,0.9,-1e2,-1e2,yi(:).',-1e2],...
   [0,0,0]),'EdgeColor','none');
plot(xi,yi,'w-');

for k = 0:8
    set(text(xy(4 + k*4,1), xy(4 + k*4,2), sprintf('%d', 460 + k*20)),'Color',[1,1,1]);
end

axis([0,0.8,0,0.9]);
xlabel('x');
ylabel('y');
title('The CIE "tongue": the region of all colors over x and y');
set(gca,'Color',[0,0,0]);
shg;
