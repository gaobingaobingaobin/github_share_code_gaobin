%2015 Gao Bin modified version of Zhang jiang's BCS_GSR_Decoder_SBI
function [reconstructed_image Map] = BCS_GSR_Decoder_SBI_Accelerated(y, Opts)
% NumRows = Opts.NumRows;
% NumCols = Opts.NumCols;
% BlockSize = Opts.BlockSize;
% IterNum = Opts.IterNum;
% OrgImg = Opts.OrgImg;
% ALSB_Thr = Opts.ALSB_Thr;
% InitImg = Opts.InitImg;

row = Opts.NumRows;
col = Opts.NumCols;
Phi = Opts.Phi;
x_org = Opts.OrgImg;
IterNum = Opts.IterNum;
x_initial = Opts.InitImg;
block_size = Opts.block_size;
Inloop = Opts.Inloop;
mu = Opts.mu;
 Cur_PSNR_pre = csnr(x_initial,x_org,0,0);
x = im2col(x_initial, [block_size block_size], 'distinct');

u = zeros(size(x));
b = zeros(size(x));

All_PSNR = zeros(1,IterNum);


ATA = Phi'*Phi;
ATy = Phi'*y;
IM = eye(size(ATA));
  alpha11=1;
  theta1=1;
   Map=[];
for i = 1:IterNum
    theta11=theta1;
    theta1=2/(i+2);
    x_hat = x;
             
    r = col2im(x_hat - b, [block_size block_size],[row col], 'distinct');  
    u00=u;
    x_bar = GSR_Solver_CS(r, Opts);
    
    x_bar = im2col(x_bar, [block_size block_size], 'distinct');
    
    u = x_bar;
    x00=x;
    for kk = 1:Inloop
        
        d = ATA*x_hat - ATy + mu*(x_hat - u - b);
        dTd = d'*d;
        G = d'*(ATA + mu*IM)*d;
        Step_Matrix = abs(dTd./G); 
        Step_length = diag(diag(Step_Matrix));
        x = x_hat - d*Step_length;
        x_hat = x;  
        
    end
    b00=b;
    b = b - (x - u);
% %     alpha11=(1+(1+4*alpha11^2)^.5);
%     x=x-( (alpha11-1)/alpha11 )*( x-x00 );
%     b=b-( (alpha11-1)/alpha11 )*( b-b00 );
alpha11=1+theta1*(1/theta11-1);
% u=alpha11*u+(1-alpha11)*u00;
x=alpha11*x+(1-alpha11)*x00;
b=alpha11*b+(1-alpha11)*b00;
   
    x_img = col2im(x, [block_size block_size],[row col], 'distinct');
    
    Cur_PSNR = csnr(x_img,x_org,0,0);
    All_PSNR(i) = Cur_PSNR;
        tolerance=norm(Cur_PSNR -Cur_PSNR_pre)/norm(Cur_PSNR_pre);
    if tolerance<Opts.stop
        break;
    end
    Cur_PSNR_pre=Cur_PSNR;
    fprintf('IterNum = %d, PSNR = %0.2f\n',i,Cur_PSNR);
     Map=[Map,Cur_PSNR];
end

reconstructed_image = col2im(x, [block_size block_size],[row col], 'distict');

