
function [RecImg Map] = BCS_ALSB_Recovery_SBI(Y, Phi, Opts)

NumRows = Opts.NumRows;
NumCols = Opts.NumCols;
BlockSize = Opts.BlockSize;
IterNum = Opts.IterNum;
OrgImg = Opts.OrgImg;
ALSB_Thr = Opts.ALSB_Thr;
InitImg = Opts.InitImg;
mu = Opts.mu;
Inloop = Opts.Inloop;
 Cur_PSNR_pre = csnr(InitImg,OrgImg,0,0);
X = im2col(InitImg, [BlockSize BlockSize], 'distinct');

U = zeros(size(X));
B = zeros(size(X));


ATA = Phi'*Phi;
ATy = Phi'*Y;
IM = eye(size(ATA));
Map=[];
for i = 1:IterNum
    
    X_hat = X;
    
    R = col2im(X_hat-B, [BlockSize BlockSize], [NumRows NumCols], 'distinct');
    
    X_bar = ALSB_Solver(R,ALSB_Thr);    
    X_bar = im2col(X_bar, [BlockSize BlockSize], 'distinct');
    
    U = X_bar;
    
    for ii = 1:Inloop
        D = ATA*X_hat - ATy + mu*(X_hat - U - B);
        DTD = D'*D;
        G = D'*(ATA + mu*IM)*D;
        Step_Matrix = abs(DTD./G); 
        Step_length = diag(diag(Step_Matrix));
        X = X_hat - D*Step_length;
        X_hat = X;  
    end
    
    B = B - (X - U);
   
    CurImg = col2im(X, [BlockSize BlockSize], [NumRows NumCols], 'distict');
     Cur_PSNR=csnr(CurImg,OrgImg,0,0);
               tolerance=norm( Cur_PSNR-Cur_PSNR_pre)/norm(Cur_PSNR_pre);
    if tolerance<Opts.stop 
        break;
    end
    Cur_PSNR_pre=Cur_PSNR;
    fprintf('IterNum = %d, PSNR = %0.2f\n',i,csnr(CurImg,OrgImg,0,0));
      
    Map=[Map,csnr(CurImg,OrgImg,0,0)];
end

RecImg = col2im(X, [BlockSize BlockSize], [NumRows NumCols], 'distict');

