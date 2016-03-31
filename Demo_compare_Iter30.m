%
% -------------------------------------------------------------------------------------------------------
% The software implemented by MatLab 2014b are included in this package.
%
% ------------------------------------------------------------------
% Requirements
% ------------------------------------------------------------------
% *) Matlab 2014b or later with installed:
% ------------------------------------------------------------------
% Version 1.0
% Author: Bin Gao
% Email:  feimaxiao123@gmail.com
% Last modified by B. Gao, Mar. 2016

%% This software package includes simulation code for the papers:
	%[1] J. Zhang, C. Zhao, D. Zhao, and W. Gao, ''Image compressive sensing recovery using adaptively learned sparsifying basis via L0 minimization,'' Signal Processing, vol.103, no.0,  pp. 114-126, Oct. 2014.
	%   For updated versions of ALSB, as well as the article on which it is based,
    %   consult: http://idm.pku.edu.cn/staff/zhangjian/ALSB/

    %[2] J. Zhang, D.B. Zhao, and W. Gao, ''Group-based sparse representation for image restoration,'' IEEE Transactions on Image Processing, vol.23, no.8,  pp. 3336-3351, Aug. 2014.
    %   For updated versions of GSR, as well as the article on which it is based,
    %   consult: http://idm.pku.edu.cn/staff/zhangjian/GSR/

%% Path Set -- Begin %%
clear
clc
CurrPath = cd;
addpath(genpath(CurrPath));
%% Path Set -- End %%

%% Parameter Set -- Begin %%
img_numbers=4;%实验图像的个数
method_numbers=4;%方法比较的个数

subrate_number=0;%为了定位三维矩阵的第三维
subrate_numbers=3;% subrate的个数
PSNR_data0=zeros(method_numbers,img_numbers,subrate_numbers);%six method,six original image,3 subrate choice
FSIM_data0=zeros(method_numbers,img_numbers,subrate_numbers);%six method,six original image,3 subrate choice
for ImgNo=5
    for Subrate = 0.3
        subrate_number=subrate_number+1;
        switch ImgNo
            case 1
                OrgName = 'House256';
            case 2
                OrgName = 'Barbara256';
            case 3
                OrgName = 'Leaves256';
            case 4
                OrgName = 'Monarch256';
            case 5
                OrgName = 'Vessels96';
        end
        
        BlockSize = 32;
        IterNum = 30;
        OrgImgName = [OrgName '.tif'];
        OrgImg = double(imread(OrgImgName));
        [NumRows NumCols] = size(OrgImg);
        
        clear Opts
        Opts = [];
        Opts.row = NumRows;% for GSR
        Opts.org = OrgImg;
        Opts.col = NumCols;% for GSR method
        %%%%%%adding computing cost time%%%
        timeSteps = nan(1,IterNum) ;
        %%%%%%adding computing cost time%%%
        if ~isfield(Opts,'OrgName')
            Opts.OrgName = OrgName;
        end
        if ~isfield(Opts,'timeSteps')
            Opts.timeSteps = timeSteps;
        end
        if ~isfield(Opts,'OrgImg')
            Opts.OrgImg = OrgImg;
        end
        
        if ~isfield(Opts,'NumRows')
            Opts.NumRows = NumRows;
        end
        
        if ~isfield(Opts,'NumCols')
            Opts.NumCols = NumCols;
        end
        
        if ~isfield(Opts,'BlockSize')
            Opts.BlockSize = BlockSize;
        end
        Opts.block_size = BlockSize;
        if ~isfield(Opts,'Subrate')
            Opts.Subrate = Subrate;
        end
        
        if ~isfield(Opts,'IterNum')
            Opts.IterNum = IterNum;
        end
        
        if ~isfield(Opts,'ALSB_Thr')
            Opts.ALSB_Thr = 8;
        end
        
        if ~isfield(Opts,'PlogFlag')
            %             Opts.mu = 0.0025;% original data
            Opts.mu = 0.005;
        end
        
        if ~isfield(Opts,'Inloop')
            Opts.Inloop = 200;
        end
        
        if ~isfield(Opts,'PlogFlag')
            Opts.PlogFlag = 1;
        end
        
        if ~isfield(Opts,'stop')%自己加上的
            Opts.stop = 40;
        end
        
        
        if ~isfield(Opts,'mu')
            Opts.mu = 2.5e-3;
        end
        
        if ~isfield(Opts,'lambda')
            Opts.lambda = 0.082;
        end
        
        if ~isfield(Opts,'Inloop')
            Opts.Inloop = 120;
        end
        %% Parameter Set -- End %%
        
        %% CS Sampling -- Begin %%
        N = BlockSize^2;
        M = round(Subrate * N);     
        
%         % randn('seed',0);
%         PhiTemp = orth(randn(N, N))';
%         Phi = PhiTemp(1:M, :);
%         save Phi
        load Phi
        Opts.Phi = Phi; %% for GSR
        X = im2col(OrgImg, [BlockSize BlockSize], 'distinct');
        
        Y = Phi * X;
        
        PSNR_data=[];
        FSIM_data=[]; % using for generate
        
        %% Initialization -- Begin %%
        %% MH method in paper " Compressed-Sensing Recovery of Images and Video using multihypothesis predictions"
        %             [X_MH X_DDWT time_DDWT] = MH_BCS_SPL_Recovery(Y, Phi, Opts);
        %             save X_MH
        %             save X_DDWT
        %             save time_DDWT
        load X_MH
        Opts.IterNum=30;%yinwei X_MH引起变化
        if ~isfield(Opts,'InitImg')
            Opts.InitImg = X_MH;
        end
        Opts.initial = X_MH;
        %% Initialization -- End %%
        
        %         fprintf('%s,rate=%0.2f\n Initial PSNR=%0.2f\n',OrgName,Subrate,csnr(Opts.InitImg ,OrgImg,0,0));
        %% CS Recovery by ALSB -- by Jian Zhang

        [RecImg_ALSB Map_ALSB time_ALSB] = BCS_ALSB_Recovery_SBI30(Y, Phi, Opts);       
        
   %%    CS Recovery by    accelerated GSR-SBM by Bin Gao
        [Rec_im_gao Map_gao time_gao]= BCS_GSR_Decoder_SBI_Accelerated30(Y, Opts);
                   
    end
end

plot(1:30,Map_ALSB(1:30),'b',1:30,Map_gao(1:30),'r');
xlabel('Iteration number');
ylabel('PSNR(dB)');
legend('method [15]','Proposed algorithm');
