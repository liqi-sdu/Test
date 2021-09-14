clear 
close all
%% 加载 mask an index 
load mask1; load mask2; load mask3; load mask4;
P(:,:,1)=mask1;P(:,:,2)=mask2;P(:,:,3)=mask3;P(:,:,4)=mask4;
% load('F:\桌面文件\多幅2\test/index.mat');
%%  Read the target image
img1=double(imread('F:\桌面文件\多幅2\image\mnist_1024/1.bmp'));   img2=double(imread('F:\桌面文件\多幅2\image\mnist_1024/2.bmp'));
img3=double(imread('F:\桌面文件\多幅2\image\mnist_1024/3.bmp'));   img4=double(imread('F:\桌面文件\多幅2\image\mnist_1024/4.bmp'));

img5=double(imread('F:\桌面文件\多幅2\image\mnist_1024/5.bmp'));   img6=double(imread('F:\桌面文件\多幅2\image\mnist_1024/6.bmp'));
img7=double(imread('F:\桌面文件\多幅2\image\mnist_1024/7.bmp'));   img8=double(imread('F:\桌面文件\多幅2\image\mnist_1024/8.bmp'));
 
img9=double(imread('F:\桌面文件\多幅2\image\mnist_1024/9.bmp'));   img10=double(imread('F:\桌面文件\多幅2\image\mnist_1024/10.bmp'));
img11=double(imread('F:\桌面文件\多幅2\image\mnist_1024/11.bmp')); img12=double(imread('F:\桌面文件\多幅2\image\mnist_1024/12.bmp'));

img13=double(imread('F:\桌面文件\多幅2\image\mnist_1024/13.bmp'));   img14=double(imread('F:\桌面文件\多幅2\image\mnist_1024/14.bmp'));
img15=double(imread('F:\桌面文件\多幅2\image\mnist_1024/15.bmp'));   img16=double(imread('F:\桌面文件\多幅2\image\mnist_1024/16.bmp'));

img17=double(imread('F:\桌面文件\多幅2\image\mnist_1024/17.bmp'));   img18=double(imread('F:\桌面文件\多幅2\image\mnist_1024/18.bmp'));
img19=double(imread('F:\桌面文件\多幅2\image\mnist_1024/19.bmp'));   img20=double(imread('F:\桌面文件\多幅2\image\mnist_1024/20.bmp'));

img21=double(imread('F:\桌面文件\多幅2\image\mnist_1024/21.bmp'));   img22=double(imread('F:\桌面文件\多幅2\image\mnist_1024/22.bmp'));
img23=double(imread('F:\桌面文件\多幅2\image\mnist_1024/23.bmp'));   img24=double(imread('F:\桌面文件\多幅2\image\mnist_1024/24.bmp'));

img25=double(imread('F:\桌面文件\多幅2\image\mnist_1024/25.bmp'));   img26=double(imread('F:\桌面文件\多幅2\image\mnist_1024/26.bmp'));
img27=double(imread('F:\桌面文件\多幅2\image\mnist_1024/27.bmp'));   img28=double(imread('F:\桌面文件\多幅2\image\mnist_1024/28.bmp'));

img29=double(imread('F:\桌面文件\多幅2\image\mnist_1024/29.bmp'));   img30=double(imread('F:\桌面文件\多幅2\image\mnist_1024/30.bmp'));
img31=double(imread('F:\桌面文件\多幅2\image\mnist_1024/31.bmp'));   img32=double(imread('F:\桌面文件\多幅2\image\mnist_1024/0.bmp'));

%% 
pro1=mask1.*img1;   pro2=mask2.*img2;    pro3=mask3.*img3;     pro4=mask4.*img4;
pro5=mask1.*img5;   pro6=mask2.*img6;    pro7=mask3.*img7;     pro8=mask4.*img8;

pro9=mask1.*img9;   pro10=mask2.*img10;  pro11=mask3.*img11;   pro12=mask4.*img12;
pro13=mask1.*img13; pro14=mask2.*img14;  pro15=mask3.*img15;   pro16=mask4.*img16;

pro17=mask1.*img17; pro18=mask2.*img18;  pro19=mask3.*img19;   pro20=mask4.*img20;
pro21=mask1.*img21; pro22=mask2.*img22;  pro23=mask3.*img11;   pro24=mask4.*img24;

pro25=mask1.*img25; pro26=mask2.*img26;  pro27=mask3.*img26;   pro28=mask4.*img28;
pro29=mask1.*img29; pro30=mask2.*img30;  pro31=mask3.*img30;   pro32=mask4.*img32;

y1=pro3+pro4+pro2+pro1;y2=pro5+pro6+pro7+pro8; y3=pro9+pro10+pro11+pro12;  y4=pro13+pro14+pro15+pro16;
y5=pro17+pro18+pro19+pro20; y6=pro21+pro22+pro23+pro24; y7=pro25+pro26+pro27+pro27+pro28; y8=pro29+pro30+pro31+pro32;
%% 生成正弦条纹
AA1=zeros(1024);AA1(385,385)=1;
BB1=real(ifft2(AA1));
CC1=BB1./max(max(BB1)); 

AA2=zeros(1024);AA2(385,129)=1;
BB2=real(ifft2(AA2));
CC2=BB2./max(max(BB2)); 
 
AA3=zeros(1024);AA3(385,897)=1;
BB3=real(ifft2(AA3));
CC3=BB3./max(max(BB3)); 

AA4=zeros(1024);AA4(385,641)=1;
BB4=real(ifft2(AA4));
CC4=BB4./max(max(BB4)); 

AA5=zeros(1024);AA5(129,385)=1;
BB5=real(ifft2(AA5));
CC5=BB5./max(max(BB5)); 

AA6=zeros(1024);AA6(129,129)=1;
BB6=real(ifft2(AA6));
CC6=BB6./max(max(BB6)); 

AA7=zeros(1024);AA7(129,897)=1;
BB7=real(ifft2(AA7));
CC7=BB7./max(max(BB7)); 

AA8=zeros(1024);AA8(129,641)=1;
BB8=real(ifft2(AA8));
CC8=BB8./max(max(BB8)); 
%% 移频
y1=y1.*CC1; 
y2=y2.*CC2; 

y3=y3.*CC3; 
y4=y4.*CC4;
y5=y5.*CC5; 
y6=y6.*CC6;
y7=y7.*CC7; 
% y8=y8.*CC8;

y_28=y1+y2+y3+y4+y5+y6+y7;
% figure(5); imshow(y)

%% 加载文件路径
image_root = 'F:\桌面文件\深度学习处理\3_4.18_网络解码_minst\minst_1024/';
image_file = dir(fullfile(image_root,'*.bmp'));

%% 开始制造数据集
for i = 1401 : 1600
    num =i*4;
    image1 = double(imread(fullfile(image_root,image_file(num).name)));
    image2 = double(imread(fullfile(image_root,image_file(num-1).name)));
    image3 = double(imread(fullfile(image_root,image_file(num-2).name)));
    image4 = double(imread(fullfile(image_root,image_file(num-3).name)));
    
    image(:,:,1)=image1;image(:,:,2)=image2;image(:,:,3)=image3;image(:,:,4)=image4;
    ref = encryption(P, image, y_28, CC8);
    
    ref_root='F:\桌面文件\深度学习处理\15_6_3_mnist_1024_gray\data\ref/';
    ref_name = num2str(i);   ref_name=strcat(ref_root,ref_name,'.mat');
    save(ref_name, 'ref');
    
    img_root='F:\桌面文件\深度学习处理\15_6_3_mnist_1024_gray\data\image/';
    img_name = num2str(i);   img_name=strcat(img_root,img_name,'.mat');
    save(img_name, 'image');
    
end