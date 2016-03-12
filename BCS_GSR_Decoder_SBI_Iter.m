
function [reconstructed_image Map] = BCS_GSR_Decoder_SBI_Iter(y, Opts)
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
Map=[];
 Cur_PSNR_pre = csnr(x_initial,x_org,0,0);
x = im2col(x_initial, [block_size block_size], 'distinct');

u = zeros(size(x));
b = zeros(size(x));

All_PSNR = zeros(1,IterNum);


ATA = Phi'*Phi;
ATy = Phi'*y;
IM = eye(size(ATA));

for i = 1:IterNum
    
    x_hat = x;
             
    r = col2im(x_hat - b, [block_size block_size],[row col], 'distinct');  
    
    x_bar = GSR_Solver_CS(r, Opts);
    
    x_bar = im2col(x_bar, [block_size block_size], 'distinct');
    
    u = x_bar;
    
    for kk = 1:Inloop
        
        d = ATA*x_hat - ATy + mu*(x_hat - u - b);
        dTd = d'*d;
        G = d'*(ATA + mu*IM)*d;
        Step_Matrix = abs(dTd./G); 
        Step_length = diag(diag(Step_Matrix));
        x = x_hat - d*Step_length;
        x_hat = x;  
        
    end
    
    b = b - (x - u);
    
   
    x_img = col2im(x, [block_size block_size],[row col], 'distinct');
    
    Cur_PSNR = csnr(x_img,x_org,0,0);
    tolerance=norm(Cur_PSNR -Cur_PSNR_pre)/norm(Cur_PSNR_pre);
    if tolerance<Opts.stop 
        break;
    end
    Cur_PSNR_pre=Cur_PSNR;
    All_PSNR(i) = Cur_PSNR;
    fprintf('IterNum = %d, PSNR = %0.2f\n',i,Cur_PSNR);
    Map=[Map, Cur_PSNR];
end

reconstructed_image = col2im(x, [block_size block_size],[row col], 'distict');

