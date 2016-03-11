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
%% cell set
i=0;
j=0;
cell6methodPSNR=cell(11,3);
cell6methodFSIM=cell(11,3);
cell6methodCPU=cell(11,3);
cell6methodREC=cell(11,3);
cell3methodMAP=cell(11,3);
PSNR6=zeros(6,1);
FSIM6=zeros(6,1);
CPU6=zeros(6,1);
rec6=cell(6,1);%恢复图像
MAP3=cell(3,1);%停止迭代前每次iteration对应的PSNR，只针对SBI，GSR-SBI和GSR-ASBI
%% Parameter Set -- Begin %%

for II=1:10
% for II=11
    i=i+1;
    %     for II=3
%         for Subrate = 0.4
    for Subrate = 0.2:0.1:0.4
        j=j+1;
        ImgNo=II;
        switch ImgNo
            case 1
                OrgName = 'House';
            case 2
                OrgName = 'Barbara';
            case 3
                OrgName = 'Leaves';
            case 4
                OrgName = 'Parrots';
            case 5
                OrgName = 'Monarch';
            case 6
                OrgName = 'Vessels';
            case 7
                OrgName = 'peppers';
            case 8
                OrgName = 'cameraman';
            case 9
                OrgName = 'boats';
            case 10
                OrgName = 'head';
                case 11
                OrgName = 'lena';
        end
        
        BlockSize = 32;
        %         IterNum = 1;
        IterNum = 1;
        
        OrgImgName = [OrgName '.tif'];
        subrate_str=num2str(Subrate);
        image_title_name=[OrgName subrate_str];
        name_SBM=[OrgName ': SBM'];
        name_GSR=[OrgName ': GSR-SBM'];
        name_our=[OrgName ': GSR-SBM with acceleration'];
        if ImgNo==10
            OrgImg1 = double(imread(OrgImgName));
            OrgImg=OrgImg1(:,:,1);
        else
            OrgImg = double(imread(OrgImgName));
        end
        
        [NumRows NumCols] = size(OrgImg);
        
        clear Opts
        Opts = [];
        Opts.row = NumRows;% for GSR
        Opts.org = OrgImg;
        Opts.col = NumCols;% for GSR method
        
        if ~isfield(Opts,'OrgName')
            Opts.OrgName = OrgName;
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
            Opts.mu = 2.5e-3;
        end
        
        if ~isfield(Opts,'Inloop')
            Opts.Inloop = 200;
        end
        
        if ~isfield(Opts,'PlogFlag')
            Opts.PlogFlag = 1;
        end
        
        if ~isfield(Opts,'stop')%自己加上的
            Opts.stop = 1e-3;
        end
        
        
        if ~isfield(Opts,'mu')
            Opts.mu = 2.5e-3;
        end
        
        if ~isfield(Opts,'lambda')
            Opts.lambda = 0.082;
        end
        
        if ~isfield(Opts,'Inloop')
            Opts.Inloop = 200;
        end
        %% Parameter Set -- End %%
        
        %% CS Sampling -- Begin %%
        N = BlockSize^2;
        M = round(Subrate * N);
        
        
        % randn('seed',0);
        PhiTemp = orth(randn(N, N))';
        Phi = PhiTemp(1:M, :);
        Opts.Phi = Phi;%%for GSR
        X = im2col(OrgImg, [BlockSize BlockSize], 'distinct');
        
        Y = Phi * X;
        %% CS Sampling -- End %%
        
        %         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% TV method%%%%%%%%%%%%%%%
        % original image
        % bbr = importdata('barbara256.tif');
        % Im = double(bbr(:,:,3));
        % ratio = .6;
        Im=OrgImg;
        % sidelength = 256;
        sidelength = NumRows;
        NN = sidelength^2;
        MM = round(Subrate*NN);
        % generate measurement matrix
        p = randperm(NN);
        picks = p(1:MM);
        for ii = 1:MM
            if picks(ii) == 1
                picks(ii) = p(MM+1);
                break;
            end
        end
        perm = randperm(NN); % column permutations allowable
        A = @(x,mode) dfA(x,picks,perm,mode);
        
        % observation
        b = A(Im(:),1);
        bavg = mean(abs(b));
        
        % add noise
        sigma = 0.04;  % noise std
        noise = randn(MM,1);
        b = b + sigma*bavg*noise;
        
        % set the optional paramaters
        clear opts
        opts.mu = 2^12;
        opts.beta = 2^6;
        opts.mu0 = 2^4;       % trigger continuation shceme
        opts.beta0 = 2^-2;    % trigger continuation shceme
        opts.maxcnt = 10;
        opts.tol_inn = 1e-3;
        opts.tol = 1E-6;
        opts.maxit = 300;
        
        % reconstruction
        t = cputime;
        [estIm, out] = TVAL3(A,b,sidelength,sidelength,opts);
        estIm = estIm - min(estIm(:));
        t = cputime - t;
        PSNR_data=[];
        FSIM_data=[];%using for generate
        PSNR_data0=zeros(6,6,3);%six method,six original image,3 subrate choice
        FSIM_data0=zeros(6,6,3);%six method,six original image,3 subrate choice
        PSNR_Rec = csnr(OrgImg,estIm,0,0);
        [FSIM_Rec, FSIMc] = FeatureSIM(estIm,OrgImg);
        PSNR_data=[PSNR_data PSNR_Rec];
        FSIM_data=[FSIM_data FSIM_Rec];
        fullscreen = get(0,'ScreenSize');
        figure('Name',image_title_name,'Position',...
            [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
        colormap(gray);
        %                subplot(3,3,1);
        %             imshow(uint8(OrgImg));
        %             title(['SADMM Recovery Result PSNR = ' num2str(PSNR_Rec) ' dB']);
        %                          title(['Original Image']);
        subplot(2,3,1);
        imshow(uint8(estIm));
        title(['\fontsize{14}(a) PSNR = ' num2str(PSNR_Rec) ' dB, ',' CPU time = ',num2str(t),' s ']);

        
        %  set(title,'FontSize',20);
        PSNR6(1)=PSNR_Rec;
        FSIM6(1)=FSIM_Rec;
        CPU6(1)=t;
        rec6{1,1}= estIm;
        % printf(['TV  PSNR = ' num2str(PSNR_Rec),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec),sprintf('\n'),sprintf(' CPU time=%d',t)]);
        % printf('MH method PSNR = %2d dB\n FSIM =%2d ',num2str(PSNR_Rec),num2str(FSIM_Rec));
        
        %% Initialization -- Begin %%
        %% MH method in paper " Compressed-Sensing Recovery of Images and
        %%%%%%%%%%%%% Video using multihypothesis predictions"
        t=cputime;
        [X_MH X_DDWT tt] = MH_BCS_SPL_Recovery(Y, Phi, Opts);
        t=cputime-t;
        PSNR_Rec = csnr(OrgImg,X_DDWT,0,0);
        [FSIM_Rec, FSIMc] = FeatureSIM(X_DDWT,OrgImg);
        PSNR_data=[PSNR_data; PSNR_Rec];
        FSIM_data=[FSIM_data; FSIM_Rec];
        PSNR_Rec0 = csnr(OrgImg,X_MH,0,0);
        [FSIM_Rec0, FSIMc0] = FeatureSIM(X_MH,OrgImg);
        PSNR_data=[PSNR_data; PSNR_Rec0];
        FSIM_data=[FSIM_data; FSIM_Rec0];
        
        
        subplot(2,3,2);
        imshow(uint8(X_DDWT));
        PSNR6(2)=PSNR_Rec;
        FSIM6(2)=FSIM_Rec;
        CPU6(2)=tt;
        rec6{2,1}=X_DDWT;
        title(['\fontsize{14}(b) PSNR = ' num2str(PSNR_Rec) ' dB, ',' CPU time = ',num2str(tt),' s ']);
        %             title(['DWT  PSNR = ' num2str(PSNR_Rec) ' dB' num2str(PSNR_Rec0)]);
        %              title(['DWT  PSNR = ' num2str(PSNR_Rec),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec),sprintf('\n'),sprintf(' CPU time=%d',tt)]);
        % printf('MH method PSNR = %2d%% dB','FSIM =%2d%% ',num2str(PSNR_Rec),num2str(FSIM_Rec));
        subplot(2,3,3);
        imshow(uint8(X_MH));
        PSNR6(3)=PSNR_Rec0;
        FSIM6(3)=FSIM_Rec0;
        CPU6(3)=t;
        rec6{3,1}=X_MH;
        title(['\fontsize{14}(c) PSNR = ' num2str(PSNR_Rec0) ' dB, ',' CPU time = ',num2str(t),' s ']);
        %             title(['MH PSNR = ' num2str(PSNR_Rec0) ' dB']);
        %             title(['MH  PSNR = ' num2str(PSNR_Rec0),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec0),sprintf('\n'),sprintf(' CPU time=%d',t)]);
        printf('MH method PSNR = %2d%% dB','FSIM =%2d%% ',num2str(PSNR_Rec0),num2str(FSIM_Rec0));
        if ~isfield(Opts,'InitImg')
            Opts.InitImg = X_MH;
        end
        Opts.initial = X_MH;
        %% Initialization -- End %%
        
        fprintf('%s,rate=%0.2f\n Initial PSNR=%0.2f\n',OrgName,Subrate,csnr(Opts.InitImg ,OrgImg,0,0));
        %% CS Recovery by ALSB -- Begin %%
        %%%%%%%%%%%%%%%%%%%%%%%SBM%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% CS Recovery by ALSB -- Begin %%
        tic;
        
        [RecImg Map3] = BCS_ALSB_Recovery_SBI(Y, Phi, Opts);
        
        TimeRec = toc;
        PSNR_Rec = csnr(OrgImg,RecImg,0,0);
        [FSIM_Rec, FSIMc] = FeatureSIM(RecImg,OrgImg);
        PSNR_data=[PSNR_data; PSNR_Rec];
        FSIM_data=[FSIM_data; FSIM_Rec];
        subplot(2,3,4);
        imshow(uint8(RecImg));
        PSNR6(4)=PSNR_Rec;
        FSIM6(4)=FSIM_Rec;
        CPU6(4)=TimeRec;
        rec6{4,1}=RecImg;
        MAP3{1,1}=Map3;
        title(['\fontsize{14}(d) PSNR = ' num2str(PSNR_Rec) ' dB, ',' CPU time = ',num2str(TimeRec),' s ']);
        % printf(['SBM  PSNR = ' num2str(PSNR_Rec),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec),sprintf('\n'),sprintf(' CPU time=%d',TimeRec)]);
        %%%%%%%%%%%%%%%%%%%%%%%%%SBM%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GSR by
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%zhangjian%%%%%%%%%%%%
        t=cputime;
        [Rec_im Map1]= BCS_GSR_Decoder_SBI_Iter(Y, Opts);
        t=cputime-t;
        PSNR_Rec = csnr(OrgImg,Rec_im,0,0);
        [FSIM_Rec, FSIMc] = FeatureSIM(Rec_im,OrgImg);
        PSNR_data=[PSNR_data; PSNR_Rec];
        FSIM_data=[FSIM_data; FSIM_Rec];
        subplot(2,3,5);
        imshow(uint8(Rec_im));
        PSNR6(5)=PSNR_Rec;
        FSIM6(5)=FSIM_Rec;
        CPU6(5)=t;
        rec6{5,1}=Rec_im;
        MAP3{2,1}=Map1;
        title(['\fontsize{14}(e) PSNR = ' num2str(PSNR_Rec) ' dB, ',' CPU time = ',num2str(t),' s ']);
        % printf(['GSR-SBM  PSNR = ' num2str(PSNR_Rec),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec),sprintf('\n'),sprintf(' CPU time=%d',t)]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GSR by
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%zhangjian%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GSR with accelerated step by
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%zhangjian%%%%%%%%%%%%
        t=cputime;
        [Rec_im Map2]= BCS_GSR_Decoder_SBI_Accelerated(Y, Opts);
        t=cputime-t;
        PSNR_Rec = csnr(OrgImg,Rec_im,0,0);
        [FSIM_Rec, FSIMc] = FeatureSIM(Rec_im,OrgImg);
        PSNR_data=[PSNR_data; PSNR_Rec];
        FSIM_data=[FSIM_data; FSIM_Rec];
        %         PSNR_data0(:,ImgNo,(Subrate*10)-1)=PSNR_data;
        %         FSIM_data0(:,ImgNo,(Subrate*10)-1)=FSIM_data;
        subplot(2,3,6);
        imshow(uint8(Rec_im));
        PSNR6(6)=PSNR_Rec;
        FSIM6(6)=FSIM_Rec;
        CPU6(6)=t;
        rec6{6,1}=Rec_im;
        MAP3{3,1}=Map2;
        title(['\fontsize{14}(f) PSNR = ' num2str(PSNR_Rec) ' dB, ',' CPU time = ',num2str(t),' s']);
        %          title(['Accelerated GSR-SBM  PSNR = ' num2str(PSNR_Rec),'dB',sprintf('\n'), 'FSIM=' num2str(FSIM_Rec),sprintf('\n'),sprintf(' CPU time=%d',t)]);
        %         dlmwrite('d:\data1.txt',PSNR_data0);%%D=dlmread('d:\data1.txt')can load data to D
        %         dlmwrite('d:\data2.txt',FSIM_data0);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GSR with accelerated step by
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%zhangjian%%%%%%%%%%%%
        
        %          PSNR6
        %         FSIM6
        %         CPU6
        cell6methodPSNR{i,j}=PSNR6;
        cell6methodFSIM{i,j}=FSIM6;
        cell6methodCPU{i,j}=CPU6;
        cell6methodREC{i,j}=rec6;
        cell3methodMAP{i,j}=MAP3;
        save cell6methodPSNR
        save cell6methodFSIM
        save cell6methodCPU
        save cell3methodMAP
                saveas(gcf,['.\result_image\',image_title_name '.fig']);%保存到当前目录下以
        
    end
end


%           load PSNR6
%         load FSIM6
%         load CPU6