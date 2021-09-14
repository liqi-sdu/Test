function [ref] = encryption(P,img, y_28, CC1)
   %%  加密
    pro1 = P(:,:,1).*img(:,:,1);
    pro2 = P(:,:,2).*img(:,:,2);
    pro3 = P(:,:,3).*img(:,:,3);
    pro4 = P(:,:,4).*img(:,:,4);
    y1 = pro1+pro2+pro3+pro4;
    y1 = y1.*CC1;
    
    y = y1 + y_28;
    %% 置乱加恢复
%     yr=zeros(1024);
%     yr(index)=y;
% %     figure(6),imshow(yr,[]);
%     % 剪裁
%     y_clipp = ones(1024);
%     % 方式1
% %     y_clipp(1:290, 1:290)=0;
%     % 方式2
% %     y_clipp(335:689, 335:689)=0;
% %     % 方式3
%      y_clipp(1:198, 1:198)=0;
%      y_clipp(1:198, 827:1024)=0;
%      y_clipp(827:1024, 1:198)=0;
%      y_clipp(827:1024, 827:1024)=0;
%     yr = yr.*y_clipp;
%     figure(7),imshow(yr)
    % 恢复
%     y=yr(index);
%     y=reshape(y,[1024 1024]);
%     figure(8),imshow(y,[]);
    
    fy = fftshift(fft2(y));
    %% 频域滤波
    [a,b]=size(fy);
    a0=385;
    b0=897;
    d=128;
    for i=1:a
        for j=1:b
            distance=sqrt((i-a0)^2+(j-b0)^2);
            if distance<=d
                h=1;
            else
                h=0;
            end
            fy(i,j)=h*fy(i,j);
        end
    end
% 取第一个亮点
Fiter2 = zeros(256);
Fiter2(1:256,1:256) = fy(257:512,769:1024);
Fiter2 = padarray(Fiter2, [384,384]);

%% 重建密文
ref = ifft2(ifftshift(Fiter2));
ref = real(ref);
% imshow(ref,[])
%% 添加噪声
% ref_max = max(max(ref));
% ref_noise = ref/ref_max;
% A=[0.05,0.06,0.07,0.08,0.1];
% g = imnoise(ref_noise, 'salt & pepper', A(ceil(rand*5)));
% ref = g*ref_max;
end
