% VP 2d, local conservation with double projection full 

clear all
%close all

%%% Bhavana Jonnalagadda (BJ)
data_out_path = "..\..\data\gen_plasma_4d\n32\"; % PC Path
% data_out_path = "../../data/gen_plasma_n64/mat_0/"; % Mac/Linux Path


a = []; % coefficient
prob = 2; % problem

order = 5;
T =23;

Nx = 32;
Nv = 64;

mm = 3;

if mm>1
    MM = mm+1;
else
    MM = mm;
end



% for truncation
opts.max_rank = Nx;
opts.rel_eps = 1e-5;

opts1.max_rank = opts.max_rank;
opts1.rel_eps = 1e-15;
%opts1.abs_eps = 1e-13;
%
problem = [];
if prob==1
    problem = 'landau2d_weak';
    a = 0.01;
    kx = 0.5;
    vmax = 6;
elseif prob ==2
    problem = 'landau2d_strong';
    a = 0.5;
    kx = 0.5;
    vmax = 6;
elseif prob == 3
    problem = 'twostream2d';
    vmax = 8;
    a = 1e-3;
    kx = 0.2;
elseif prob == 4
    problem = 'twostream1dii';
    a = 0.01;
end


if order == 5
    CWENO = @CWENO5;
else
    CWENO = @CWENO9;
end

kt = 2; % temporal order
Lx = 2*pi/kx;
Lv = vmax * 2; % [-6,6]
kv = 2*pi/Lv;




hx = Lx/Nx;
hv = Lv/Nv;


dt = min(hx,hv)/10;
Nt = round(T/dt);
dt = T/Nt;



x = hx*(1:Nx)';
v = -vmax -0.5*hv+ hv*(1:Nv)';


ex = ones(Nx,1);
ev = ones(Nv,1);



% for computing rho via contraction

htrho =  htensor({ones(Nv,1), ones(Nv,1)});

htj1 =  htensor({v, ones(Nv,1)});
htj2 =  htensor({ones(Nv,1), v});

hte = htensor({v.^2, v.^2});
% hte1 = htensor({v.^2, ones(Nv,1)});
% hte2 = htensor({ones(Nv,1), v.^2});
htk = htensor({[ones(Nx,1) ones(Nx,1)], [ones(Nx,1) ones(Nx,1)],...
               [v.^2 ones(Nv,1)], [ones(Nv,1) v.^2]});

htj1s = htensor({ones(Nx,1), ones(Nx,1), v, ones(Nv,1)});
htj2s = htensor({ones(Nx,1), ones(Nx,1), ones(Nv,1), v});
           
% Fourier derivatives
DNx = kx*1i*[0:Nx/2-1 0 -Nx/2+1:-1]'; % for first derivative
DNv = kv*1i*[0:Nv/2-1 0 -Nv/2+1:-1]'; % for first derivative

DNx2 = DNx.^2;
DNv2 = DNv.^2;

% for poisson
D = DNx.^2*ones(Nx,1)' + ones(Nx,1)*(DNx.^2)';
D(1,1) = 1;
D(Nx/2+1,Nx/2+1) = 1;
D(Nx/2+1,1) = 1;
D(1,Nx/2+1) = 1;

D1 = ones(Nx,1)*DNx.'./D; % .' means transpose without conjugate
D2 = DNx*ones(Nx,1)'./D;


%
if prob == 1  ||  prob == 2
    
    f = htensor({[1/(2*pi)*( a*cos(kx*x)), ones(Nx,1), 1/sqrt(2*pi)*ones(Nx,1)], ...
        [ones(Nx,1), 1/(2*pi)* a*cos(kx*x), 1/sqrt(2*pi)*ones(Nx,1)], [exp(-v.^2/2), exp(-v.^2/2), exp(-v.^2/2)], ...
        [exp(-v.^2/2), exp(-v.^2/2), exp(-v.^2/2)]}); % create a htd tensor from CP format
    
elseif prob ==3
    v0 = 2.4;
    f = htensor({[0.25/(2*pi)*ones(Nx,1), 0.25/(2*pi)*( a*cos(kx*x)), ones(Nx,1)], ...
        [ones(Nx,1), ones(Nx,1), 0.25/(2*pi)*( a*cos(kx*x))],...
        [exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2), exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2), exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2)], ...
        [exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2), exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2), exp(-(v-v0).^2/2)+exp(-(v+v0).^2/2)]}); % create a htd tensor from CP format    
end



% f = htenrandn([Nx Nx Nv Nv]) +  htenrandn([Nx Nx Nv Nv]) +  htenrandn([Nx Nx Nv Nv])...
%      +  htenrandn([Nx Nx Nv Nv]) + htenrandn([Nx Nx Nv Nv]);
% 


rk = rank(f);


mass = zeros(Nt,2);
e_rank = zeros(Nt, size(rk,2));
energy = zeros(Nt,2);

% 

Vt = -vmax^2/log(opts.rel_eps);

% for weighted SVD truncation
if prob == 1 || prob == 2 || prob == 3
    Weight = @(v) exp(-v.^2/Vt);
elseif prob >= 4
    return
    %Weight = @(v) (1+5*v.^2).*exp(-v.^2/2);   
end


vw = Weight(v);

vwh = sqrt(vw);

ivwh = 1./vwh;

ivw = 1./vw;

econst = sum(vw.*v.^2)/sum(vw);

ee = v.^2-econst;

evw = ee.*vw;
vvw = v.*vw;


htki = htensor({[ee ev], [ev ee]});
% 

% 

%norm1 = [sum(vw), sum(v.^2.*vw), sum((v.^2-econst).^2.*vw)];

if mm == 1
    
    uu = {htrho};

    Norm = sum(vw)^2;

    B1 = 1;
    B3 = 1;
    U6 = vw;
    U7 = vw;
    u6 = ev;
    u7 = u6;
    
elseif mm == 2
   

    uu = {htrho, htj1, htj2};


    Norm = [sum(vw)^2, sum(v.^2.*vw)*sum(vw), sum(v.^2.*vw)*sum(vw)];
    
    B1 = eye(3);
    B3 = zeros(2,2,3);
    B3(1,1,1) = 1;
    B3(2,1,2) = 1;
    B3(1,2,3) = 1;
    U6 = [vw vvw];
    U7 = [vw vvw];
    
    u6 = [ev v];
    u7 = u6;
    

    
    
elseif mm == 3
    

    uu = {htrho, htj1, htj2, htki};


    Norm = [sum(vw)^2, sum(v.^2.*vw)*sum(vw), sum(v.^2.*vw)*sum(vw), ...
        2*sum((v.^2-econst).^2.*vw)*sum(vw)];
    
    B1 = eye(4);
    B3 = zeros(3,3,4);
    B3(1,1,1) = 1;
    B3(2,1,2) = 1;
    B3(1,2,3) = 1;
    B3(3,1,4) = 1;
    B3(1,3,4) = 1;
    
    U6 = [vw vvw evw];
    U7 = [vw vvw evw];
    
    u6 = [ev v ee];
    u7 = u6;
    
    
end

macro = cell(MM,1);

bn = zeros(2,MM+1);
bn(1,1) = 0;
bn(2,1) = 0;
for m = 1:MM

    mt = squeeze(ttt(f, uu{m}, [3 4], [1 2])); % macro denotes the unnormalized rho
    mt = 1/Norm(m)*truncate_std(mt, opts1); % normalized macro
    macro{m} = mt;
    rk = rank(mt);
    bn(1,m+1) = bn(1,m) + rk(2);
    bn(2,m+1) = bn(2,m) + rk(3);

end

switch mm
    case 1
        U3 = macro{1}.U{2};
        U4 = macro{1}.U{3};
        
        B2 = macro{1}.B{1};

    case 2
        U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2}];
        U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3}];
        
        B2 = zeros(bn(1,end),bn(2,end),3);
        B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
        B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
        B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};
      
    case 3
        U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2} macro{4}.U{2}];
        U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3} macro{4}.U{3}];
        
        B2 = zeros(bn(1,end),bn(2,end),4);
        B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
        B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
        B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};
        B2(bn(1,4)+1:bn(1,5),bn(2,4)+1:bn(2,5),4) = macro{4}.B{1};
        
        
end


UU = {[],[],[], U3, U4, U6, U7};
BB = {B1, B2, B3, [], [], [], []};
f1 = htensor(f.children, f.dim2ind, UU, BB, false);

% rho = full(ttt(f, htrho, [3 4], [1 2]))*hv^2;
% rho1 =  full(ttt(f1, htrho, [3 4], [1 2]))*hv^2;
% 
% 
% norm(rho-rho1)
% % 
% J1 = full(ttt(f, htj1, [3 4], [1 2]))*hv^2;
% J11 =  full(ttt(f1, htj1, [3 4], [1 2]))*hv^2;
% 
% J2 = full(ttt(f, htj2, [3 4], [1 2]))*hv^2;
% J21 =  full(ttt(f1, htj2, [3 4], [1 2]))*hv^2;
% 
% 
% max(abs(J1-J11),[],'all')
% max(abs(J2-J21),[],'all')
% % 
% ek = 0.5*hx^2*hv^2 * ttt(f, htk, [1 2 3 4]);
% ek1 = 0.5*hx^2*hv^2 * ttt(f1, htk, [1 2 3 4]);
% 
% (ek-ek1)/ek
% 
% return 


f1w = ttm(f1, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);

f2 = f - f1;

F2 = ttm(f2, {@(y) ivw.*y, @(y) ivw.*y}, [3, 4]); %????? Many questions
    
innerprod(F2,f1)

% create a projection


rho = full(ttt(f, htrho, [3 4], [1 2]))*hv^2;
rho1 =  full(ttt(f1, htrho, [3 4], [1 2]))*hv^2;

% total averaged mass
tmass = sum(rho,'all') * hx^2 / Lx^2;
%tmass1 = sum(rho1,'all') * hx^2 / Lx^2;

fi_sum = full(ttv(f, {ex,ex,ev,ev}, [1 2 3 4])); 

rho = rho - tmass;


%



%



rhohat = fft2(rho);

%     contour(xx,yy,rhof);
%
%     pause


E2hat = rhohat.*D1;
E1hat = rhohat.*D2;
E1 = real(ifft2(E1hat));
E2 = real(ifft2(E2hat));

[xx, yy] = meshgrid(x,x);
% surf(xx,yy,E2);
% colorbar;



ener_e = 0.5*hx^2 * (norm(E1,'fro')^2 + norm(E2,'fro')^2);
ener_k = 0.5*hx^2*hv^2 * ttt(f, htk, [1 2 3 4]);
mom1_s = full(ttv(f, {ex,ex,v,ev}, [1 2 3 4]))*hv^2*hx^2; 
mom2_s = full(ttv(f, {ex,ex,ev,v}, [1 2 3 4]))*hv^2*hx^2; 


ener_E = zeros(Nt,2);
mom1 = zeros(Nt,2);
mom2 = zeros(Nt,2);


mass(1,:) = [0, 0];
rk = rank(f);
e_rank(1,:) = [0, rk(2:end)];
ener_E(1,:) = [0, ener_e];
energy(1,:) = [0, ener_e + ener_k];
mom1(1,:) = [0, mom1_s];
mom2(1,:) = [0, mom2_s];


% if ~f.is_orthog()
%     f = orthog(f);
% end





tic;

tn = 0;
list = cell(3,1);
list{1} = f;




close all

for i=2:3
    
    
    Ux1 = f.U{4};
    Ux2 = f.U{5};
    Uv1 = f.U{6};
    Uv2 = f.U{7};
    
    DUx1m = zeros(size(Ux1));
    DUx2m = zeros(size(Ux2));
    DUv1m = zeros(size(Uv1));
    DUv2m = zeros(size(Uv2));
    
    DUx1p = zeros(size(Ux1));
    DUx2p = zeros(size(Ux2));
    DUv1p = zeros(size(Uv1));
    DUv2p = zeros(size(Uv2));
    
    for j=1:size(Ux1,2)
        
        [du, ju] = CWENO(Ux1(:,j),hx);
        DUx1m(:,j) = du;% positive velocity
        DUx1p(:,j) = ju;
    end
    
    %     beta = max(abs(E));
    
    for j=1:size(Ux2,2)
        
        [du, ju] = CWENO(Ux2(:,j),hx);
        DUx2m(:,j) = du;% positive velocity
        DUx2p(:,j) = ju;
    end
    
    for j=1:size(Uv1,2)
        
        [du, ju] = CWENO(Uv1(:,j),hv);
        DUv1m(:,j) = du;% positive velocity
        DUv1p(:,j) = ju;
    end
    
    for j=1:size(Uv2,2)
        
        [du, ju] = CWENO(Uv2(:,j),hv);
        DUv2m(:,j) = du;% positive velocity
        DUv2p(:,j) = ju;
    end
    
    
    
    fv1x1m = f;
    fv1x1p = f;
    fv2x2m = f;
    fv2x2p = f;
    
    fv1x1m.U{4} = DUx1m;
    fv1x1p.U{4} = DUx1p;
    fv2x2m.U{5} = DUx2m;
    fv2x2p.U{5} = DUx2p;
    
    % [x1,x2,v1,v2] = [1 2 3 4]
    fv1x1m = -dt*ttm(fv1x1m, @(y) max(v,0).*y, 3);
    fv1x1p = -dt*ttm(fv1x1p, @(y) min(v,0).*y, 3);
    fv2x2m = -dt*ttm(fv2x2m, @(y) max(v,0).*y, 4);
    fv2x2p = -dt*ttm(fv2x2p, @(y) min(v,0).*y, 4);
    
    E1max = max(abs(E1(:)));
    E2max = max(abs(E2(:)));
%     
    E1m = htensor.truncate_ltr(0.5*E1 + 0.5*E1max*ones(Nx,Nx), opts);
    E1p = htensor.truncate_ltr(0.5*E1 - 0.5*E1max*ones(Nx,Nx), opts);
    E2m = htensor.truncate_ltr(0.5*E2 + 0.5*E2max*ones(Nx,Nx), opts);
    E2p = htensor.truncate_ltr(0.5*E2 - 0.5*E2max*ones(Nx,Nx), opts);
    

%     
%     E1m = htensor.truncate_ltr(max(E1,0), opts);
%     E1p = htensor.truncate_ltr(min(E1,0), opts);
%     E2m = htensor.truncate_ltr(max(E2,0), opts);
%     E2p = htensor.truncate_ltr(min(E2,0), opts);

    
    
    fe1v1m = -dt*Etof2d(f, E1m);
    fe1v1p = -dt*Etof2d(f, E1p);
    fe2v2m = -dt*Etof2d(f, E2m);
    fe2v2p = -dt*Etof2d(f, E2p);
    
    fe1v1m.U{6} = DUv1m;
    fe1v1p.U{6} = DUv1p;
    
    fe2v2m.U{7} = DUv2m;
    fe2v2p.U{7} = DUv2p;
    
   
    
    %fsum = {f, fv1x1m, fv1x1p, fv2x2m, fv2x2p, fe1v1m, fe1v1p, fe2v2m, fe2v2p};
    fsum = f + fv1x1m + fv1x1p + fv2x2m + fv2x2p + fe1v1m + fe1v1p + fe2v2m...
        + fe2v2p;
    
%     macro = ttt(fsum, uu, [3 4], [1 2]); % macro denotes the unnormalized rho
%     macro = 1/Norm*truncate_std(macro, opts1); % truncate and normalized the solution
% 
%     f1 = ttt(macro, U, 3, 3);
%     UU = {[],[],[], f1.U{3}, f1.U{4}, f1.U{6}, f1.U{7}};
%     BB = {f1.B{1}, f1.B{2}, f1.B{5}, [], [], [], []};
%     f1 = htensor(f.children, f.dim2ind, UU, BB, false);
%     f1w = ttm(f1, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    


    bn = zeros(2,MM+1);
    bn(1,1) = 0;
    bn(2,1) = 0;
    for m = 1:MM

        mt = squeeze(ttt(fsum, uu{m}, [3 4], [1 2])); % macro denotes the unnormalized rho
        mt = 1/Norm(m)*truncate_std(mt, opts1); % normalized macro
        macro{m} = mt;
        rk = rank(mt);
        bn(1,m+1) = bn(1,m) + rk(2);
        bn(2,m+1) = bn(2,m) + rk(3);

    end

    switch mm
        case 1
            U3 = macro{1}.U{2};
            U4 = macro{1}.U{3};

            B2 = macro{1}.B{1};

        case 2
            U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2}];
            U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3}];

            B2 = zeros(bn(1,end),bn(2,end),3);
            B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
            B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
            B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};

        case 3
            U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2} macro{4}.U{2}];
            U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3} macro{4}.U{3}];

            B2 = zeros(bn(1,end),bn(2,end),4);
            B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
            B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
            B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};
            B2(bn(1,4)+1:bn(1,5),bn(2,4)+1:bn(2,5),4) = macro{4}.B{1};


    end


    UU = {[],[],[], U3, U4, U6, U7};
    BB = {B1, B2, B3, [], [], [], []};
    f1 = htensor(f.children, f.dim2ind, UU, BB, false);
    f1w = ttm(f1, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);


    
    
    fw = ttm(f, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv1x1mw = ttm(fv1x1m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv1x1pw = ttm(fv1x1p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv2x2mw = ttm(fv2x2m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv2x2pw = ttm(fv2x2p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe1v1mw = ttm(fe1v1m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe1v1pw = ttm(fe1v1p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe2v2mw = ttm(fe2v2m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe2v2pw = ttm(fe2v2p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    %, fv1x1p, fv2x2m, fv2x2p, fe1v1m, fe1v1p, fe2v2m, fe2v2p
    
    f2wsum = {fw,fv1x1mw, fv1x1pw, fv2x2mw, fv2x2pw, fe1v1mw, fe1v1pw, fe2v2mw,...
        fe2v2pw, -f1w};
    
    f2w_t = htensor.truncate_sum(f2wsum, opts);
    f2 = ttm(f2w_t, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    
    % the key is to treat BB3
    rk = rank(f2);
    f2B3 = f2.B{3};
    f2B3n = zeros(rk(6)+mm,rk(7)+mm,rk(3));
    f2B3n(1:rk(6),1:rk(7),:) = f2B3;

    
    a6 = u6'* f2.U{6};
    a7 = u7'* f2.U{7};
    
    f2B3 = reshape(f2B3,rk(6)*rk(7),[]);
    b3 = -1/Norm(1)*(kron(a7(1,:),a6(1,:))* f2B3);
    f2B3n(rk(6)+1,rk(7)+1,:) = b3;
    
    if mm>1
        b3 = -1/Norm(2)*(kron(a7(1,:),a6(2,:))* f2B3);
        f2B3n(rk(6)+2,rk(7)+1,:) = b3;
        b3 = -1/Norm(3)*(kron(a7(2,:),a6(1,:))* f2B3);
        f2B3n(rk(6)+1,rk(7)+2,:) = b3;
        if mm>2
            b3 = -1/Norm(4)*((kron(a7(1,:),a6(3,:))+kron(a7(3,:),a6(1,:)))* f2B3);
            f2B3n(rk(6)+3,rk(7)+1,:) = b3;
            f2B3n(rk(6)+1,rk(7)+3,:) = b3;
        end
        
    end
        

    UU = {[],[],[], f2.U{4}, f2.U{5}, [f2.U{6} U6], [f2.U{7} U7]};
    BB = {f2.B{1}, f2.B{2}, f2B3n, [], [], [], []};
    f2o = htensor(f2.children, f2.dim2ind, UU, BB, false);
    
    f = f1 + f2o;

    
    
    ft_sum = full(ttv(f, {ex,ex,ev,ev}, [1 2 3 4])); 
    disp("relative mass after truncation:");
    mass_err = 1 - ft_sum/fi_sum;
    disp(mass_err);
    
%     rho1 = full(ttt(f1, htrho, [3 4], [1 2]))*hv^2;
%     
%     mass1 = sum(rho1,'all') * hx^2 - tmass*Lx^2;
%     
%     rho2 = full(ttt(f2, htrho, [3 4], [1 2]))*hv^2;
%     
%     mass2 = sum(rho2,'all') * hx^2;
    
    list{i} = f;
    
    rk = rank(f);
    
    tn = tn + dt;
    
    
    rho = full(ttt(f, htrho, [3 4], [1 2]))*hv^2 - tmass;
    
    rhohat = fft2(rho);

    E2hat = rhohat.*D1;
    E1hat = rhohat.*D2;
    E1 = real(ifft2(E1hat));
    E2 = real(ifft2(E2hat));
    
    
    ener_e = 0.5*hx^2 * (norm(E1,'fro')^2 + norm(E2,'fro')^2);
    ener_k = 0.5*hx^2*hv^2 * ttt(f, htk, [1 2 3 4]);
    mom1_s = full(ttv(f, {ex,ex,v,ev}, [1 2 3 4]))*hv^2*hx^2; 
    mom2_s = full(ttv(f, {ex,ex,ev,v}, [1 2 3 4]))*hv^2*hx^2; 
    
    disp([tn, rk]);
    
    mass(i,:) = [tn, mass_err];
    e_rank(i,:) = [tn, rk(2:end)];
    ener_E(i,:) = [tn, ener_e];
    energy(i,:) = [tn, ener_e + ener_k];
    mom1(i,:) = [tn, mom1_s];
    mom2(i,:) = [tn, mom2_s];
    


    
    
    
    
end


for i = 4:Nt+1

    %%%% Bhavana Jonnalagadda (BJ)
    % Save f only, not the decomposition
    full_f = full(f);
    h5create(data_out_path + (i-4) + ".hdf5","/f", size(full_f))
    h5write(data_out_path + (i-4) + ".hdf5","/f", full_f)
    
    Ux1 = f.U{4};
    Ux2 = f.U{5};
    Uv1 = f.U{6};
    Uv2 = f.U{7};
    
    DUx1m = zeros(size(Ux1));
    DUx2m = zeros(size(Ux2));
    DUv1m = zeros(size(Uv1));
    DUv2m = zeros(size(Uv2));
    
    DUx1p = zeros(size(Ux1));
    DUx2p = zeros(size(Ux2));
    DUv1p = zeros(size(Uv1));
    DUv2p = zeros(size(Uv2));
    
    for j=1:size(Ux1,2)
        
        [du, ju] = CWENO(Ux1(:,j),hx);
        DUx1m(:,j) = du;% positive velocity
        DUx1p(:,j) = ju;
    end
    
    %     beta = max(abs(E));
    
    for j=1:size(Ux2,2)
        
        [du, ju] = CWENO(Ux2(:,j),hx);
        DUx2m(:,j) = du;% positive velocity
        DUx2p(:,j) = ju;
    end
    
    for j=1:size(Uv1,2)
        
        [du, ju] = CWENO(Uv1(:,j),hv);
        DUv1m(:,j) = du;% positive velocity
        DUv1p(:,j) = ju;
    end
    
    for j=1:size(Uv2,2)
        
        [du, ju] = CWENO(Uv2(:,j),hv);
        DUv2m(:,j) = du;% positive velocity
        DUv2p(:,j) = ju;
    end
    
    
    
    fv1x1m = f;
    fv1x1p = f;
    fv2x2m = f;
    fv2x2p = f;
    
    fv1x1m.U{4} = DUx1m;
    fv1x1p.U{4} = DUx1p;
    fv2x2m.U{5} = DUx2m;
    fv2x2p.U{5} = DUx2p;
    
    % [x1,x2,v1,v2] = [1 2 3 4]
    fv1x1m = -1.5*dt*ttm(fv1x1m, @(y) max(v,0).*y, 3);
    fv1x1p = -1.5*dt*ttm(fv1x1p, @(y) min(v,0).*y, 3);
    fv2x2m = -1.5*dt*ttm(fv2x2m, @(y) max(v,0).*y, 4);
    fv2x2p = -1.5*dt*ttm(fv2x2p, @(y) min(v,0).*y, 4);
    
    E1max = max(abs(E1(:)));
    E2max = max(abs(E2(:)));
    
%     
    E1m = htensor.truncate_ltr(0.5*E1 + 0.5*E1max*ones(Nx,Nx), opts);
    E1p = htensor.truncate_ltr(0.5*E1 - 0.5*E1max*ones(Nx,Nx), opts);
    E2m = htensor.truncate_ltr(0.5*E2 + 0.5*E2max*ones(Nx,Nx), opts);
    E2p = htensor.truncate_ltr(0.5*E2 - 0.5*E2max*ones(Nx,Nx), opts);
    

    
%     E1m = htensor.truncate_ltr(max(E1,0), opts);
%     E1p = htensor.truncate_ltr(min(E1,0), opts);
%     E2m = htensor.truncate_ltr(max(E2,0), opts);
%     E2p = htensor.truncate_ltr(min(E2,0), opts);
    
    
    
    fe1v1m = -1.5*dt*Etof2d(f, E1m);
    fe1v1p = -1.5*dt*Etof2d(f, E1p);
    fe2v2m = -1.5*dt*Etof2d(f, E2m);
    fe2v2p = -1.5*dt*Etof2d(f, E2p);
    
    fe1v1m.U{6} = DUv1m;
    fe1v1p.U{6} = DUv1p;
    
    fe2v2m.U{7} = DUv2m;
    fe2v2p.U{7} = DUv2p;
    
    
    
    fsum = {0.25*list{1}, 0.75*f, fv1x1m, fv1x1p, fv2x2m, fv2x2p, fe1v1m, fe1v1p, fe2v2m, fe2v2p};
    
%     fsum = 0.25*list{1} + 0.75*f + fv1x1m + fv1x1p + fv2x2m + fv2x2p + fe1v1m + fe1v1p + fe2v2m...
%         + fe2v2p;

    
    
    
%     macro = ttt(fsum, uu, [3 4], [1 2]); % macro denotes the unnormalized rho
%     macro = 1/Norm*truncate_std(macro, opts1); % truncate and normalized the solution
% 
%     f1 = ttt(macro, U, 3, 3);
%     UU = {[],[],[], f1.U{3}, f1.U{4}, f1.U{6}, f1.U{7}};
%     BB = {f1.B{1}, f1.B{2}, f1.B{5}, [], [], [], []};
%     f1 = htensor(f.children, f.dim2ind, UU, BB, false);
%     f1w = ttm(f1, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
%     
    
    macro_sum = cell(size(fsum));
    bn = zeros(2,MM+1);
    bn(1,1) = 0;
    bn(2,1) = 0;
    for m = 1:MM
        switch m
            case 1
                for k = 1:length(fsum)
                    macro_sum{k} = ttv(fsum{k}, {ev ev}, [3 4]);
                end
            case 2
                for k = 1:length(fsum)
                    macro_sum{k} = ttv(fsum{k}, {v ev}, [3 4]);
                end
            case 3
                for k = 1:length(fsum)
                    macro_sum{k} = ttv(fsum{k}, {ev v}, [3 4]);
                end
            case 4
                for k = 1:length(fsum)
                    macro_sum{k} = ttv_energy(fsum{k}, ev, ee);
                end
        end            
                
        mt = 1/Norm(m)*htensor.truncate_sum(macro_sum, opts1);
        

        %mt = squeeze(ttt(f, uu{m}, [3 4], [1 2])); % macro denotes the unnormalized rho
        %mt = 1/Norm(m)*truncate_std(mt, opts1);
        macro{m} = mt;
        rk = rank(mt);
        bn(1,m+1) = bn(1,m) + rk(2);
        bn(2,m+1) = bn(2,m) + rk(3);
        %macro{m} = 1/Norm(m)*htensor.truncate_sum(macro_sum, opts1);
    end


    switch mm
        case 1
            U3 = macro{1}.U{2};
            U4 = macro{1}.U{3};

            B2 = macro{1}.B{1};

        case 2
            U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2}];
            U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3}];

            B2 = zeros(bn(1,end),bn(2,end),3);
            B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
            B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
            B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};

        case 3
            U3 = [macro{1}.U{2} macro{2}.U{2} macro{3}.U{2} macro{4}.U{2}];
            U4 = [macro{1}.U{3} macro{2}.U{3} macro{3}.U{3} macro{4}.U{3}];

            B2 = zeros(bn(1,end),bn(2,end),4);
            B2(1:bn(1,2),1:bn(2,2),1) = macro{1}.B{1};
            B2(bn(1,2)+1:bn(1,3),bn(2,2)+1:bn(2,3),2) = macro{2}.B{1};
            B2(bn(1,3)+1:bn(1,4),bn(2,3)+1:bn(2,4),3) = macro{3}.B{1};
            B2(bn(1,4)+1:bn(1,5),bn(2,4)+1:bn(2,5),4) = macro{4}.B{1};


    end


    UU = {[],[],[], U3, U4, U6, U7};
    BB = {B1, B2, B3, [], [], [], []};
    f1 = htensor(f.children, f.dim2ind, UU, BB, false);
    f1w = ttm(f1, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    
%     if tn>0.90000000000000 && tn<0.9+dt
%         disp(tn)
%         [ff1, ff2, FF2] = w_test(fsum,Nx,Nv,vw);
%     end

    
    flistw =  ttm(list{1}, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fw =      ttm(f, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv1x1mw = ttm(fv1x1m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv1x1pw = ttm(fv1x1p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv2x2mw = ttm(fv2x2m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fv2x2pw = ttm(fv2x2p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe1v1mw = ttm(fe1v1m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe1v1pw = ttm(fe1v1p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe2v2mw = ttm(fe2v2m, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    fe2v2pw = ttm(fe2v2p, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    
    
    f2wsum = {0.25*flistw, 0.75*fw, fv1x1mw, fv1x1pw, fv2x2mw, fv2x2pw, fe1v1mw, fe1v1pw, fe2v2mw,...
        fe2v2pw, -f1w};
    
    
    
%     if tn>0.90000000000000 && tn<0.9+dt
%         disp(tn)
%         f2sum = 0.25*flistw + 0.75*fw + fv1x1mw + fv1x1pw+ fv2x2mw+ fv2x2pw+ fe1v1mw+ fe1v1pw+ fe2v2mw ...
%         + fe2v2pw - f1w; 
%         ff2w_t = truncate_std(f2sum, opts);
%         fff2 = ttm(ff2w_t, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
%         
%         
%     end
    
    f2w_t = htensor.truncate_sum(f2wsum, opts);
    
    
    f2 = ttm(f2w_t, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);    



    % the key is to treat BB3
    rk = rank(f2);
    f2B3 = f2.B{3};
    f2B3n = zeros(rk(6)+mm,rk(7)+mm,rk(3));
    f2B3n(1:rk(6),1:rk(7),:) = f2B3;

    
    a6 = u6'* f2.U{6};
    a7 = u7'* f2.U{7};
    
    f2B3 = reshape(f2B3,rk(6)*rk(7),[]);
    b3 = -1/Norm(1)*(kron(a7(1,:),a6(1,:))* f2B3);
    f2B3n(rk(6)+1,rk(7)+1,:) = b3;
    
    if mm>1
        b3 = -1/Norm(2)*(kron(a7(1,:),a6(2,:))* f2B3);
        f2B3n(rk(6)+2,rk(7)+1,:) = b3;
        b3 = -1/Norm(3)*(kron(a7(2,:),a6(1,:))* f2B3);
        f2B3n(rk(6)+1,rk(7)+2,:) = b3;
        if mm>2
            b3 = -1/Norm(4)*((kron(a7(1,:),a6(3,:))+kron(a7(3,:),a6(1,:)))* f2B3);
            f2B3n(rk(6)+3,rk(7)+1,:) = b3;
            f2B3n(rk(6)+1,rk(7)+3,:) = b3;
        end
        
    end
        

    UU = {[],[],[], f2.U{4}, f2.U{5}, [f2.U{6} U6], [f2.U{7} U7]};
    BB = {f2.B{1}, f2.B{2}, f2B3n, [], [], [], []};
    f2o = htensor(f2.children, f2.dim2ind, UU, BB, false);
    
    f = f1 + f2o;
    
    
    rk = rank(f2o);
%     
%     
    err = 0;
    for k = 1:rk(3)
        f2oB3 = f2o.B{3}(:,:,k);
        err = err + kron(ev, ev)' * kron(f2o.U{7}, f2o.U{6}) * f2oB3(:);
    end
    
    
    disp('f2o B3 orthogonal error')
    disp(err)

 
    ft_sum = full(ttv(f, {ex,ex,ev,ev}, [1 2 3 4])); 
    f2o_sum = full(ttv(f2o, {ex,ex,ev,ev}, [1 2 3 4])); 
    mass_err = 1 - ft_sum/fi_sum;
    disp("relative mass after truncation:")
    disp(mass_err);
    disp("relative f2 sum:")
    err_f2 = f2o_sum/ft_sum;
    disp(f2o_sum/ft_sum);
    
%     if abs(err_f2) > 1e-10
%         disp('why the f2 error is so large?');
%         [ff1, ff2, FF2] = w_test(fsum,Nx,Nv,vw);
%     end
    
    %f = htensor.truncate_sum(fsum, opts);
    
    %     ff = full(f);
    %
    %     contourf(xx1,xx2,reshape(ff(2,2,:,:),[Nx Nx]));
    %
    %     drawnow
    
    
    
    
    list{1} = list{2};
    list{2} = list{3};
    list{3} = f;
    
    rk = rank(f);
    
    tn = tn + dt;
    
    
    rho = full(ttt(f, htrho, [3 4], [1 2]))*hv^2 - tmass;
    
    rhohat = fft2(rho);

    E2hat = rhohat.*D1;
    E1hat = rhohat.*D2;
    E1 = real(ifft2(E1hat));
    E2 = real(ifft2(E2hat));
    
    
    ener_e = 0.5*hx^2 * (norm(E1,'fro')^2 + norm(E2,'fro')^2);
    ener_k = 0.5*hx^2*hv^2 * ttt(f, htk, [1 2 3 4]);
    mom1_s = full(ttv(f, {ex,ex,v,ev}, [1 2 3 4]))*hv^2*hx^2; 
    mom2_s = full(ttv(f, {ex,ex,ev,v}, [1 2 3 4]))*hv^2*hx^2; 
    
    disp([tn, rk]);
    
    mass(i,:) = [tn, mass_err];
    e_rank(i,:) = [tn, rk(2:end)];
    ener_E(i,:) = [tn, ener_e];
    energy(i,:) = [tn, ener_e + ener_k];
    mom1(i,:) = [tn, mom1_s];
    mom2(i,:) = [tn, mom2_s];
    
    %disp(e_rank(i,:));
    
    
    
end

toc

figure;
semilogy(ener_E(1:Nt+1,1),ener_E(1:Nt+1,2));


figure

semilogy(mass(:,1), abs(mass(:,2)));


figure

semilogy(energy(2:end,1), abs(1 - energy(2:end,2)/energy(1,2)));

figure

plot(mom1(:,1),mom1(:,2));
hold on
plot(mom2(:,1),mom2(:,2));


figure;


% for computing rho via contraction

plot((1:Nt+1)'*dt, e_rank(:,2:end));

% ylim([0 4])
% yticks(0:1:4)

xlim([0 T])

xlabel('time');
ylabel('hierarchical ranks');


legend('r_{12}', 'r_{34}', 'r_1', 'r_2','r_3','r_4');

name1 = strcat(problem,'_conser_','rank_T',num2str(T),'_',num2str(Nx),'_',num2str(opts.rel_eps),'.mat');
e1_rank = [(1:Nt+1)'*dt, e_rank];
save(name1, 'e1_rank');

figure

cx = zeros(Nx,1);
cv = zeros(Nv,1);

cx(Nx/2) = 1;
cv(Nv/2) = 1;

hpv2 = htensor({cx, cv});

fxv1 = full(ttt(f,hpv2,[2 4],[1 2]));

[xx, vv] = meshgrid(x,v);

contourf(xx,vv,max(fxv1,0)','LineColor','none');
xlabel('x_1')
ylabel('v_1')
%ylim([0 0.14])

colorbar;

figure
hpv3 = htensor({cx, cx});
fxv2 = full(ttt(f,hpv3,[1 2],[1 2]));
[vv1, vv2] = meshgrid(v,v);

contourf(vv1,vv2,max(fxv2,0)','LineColor','none');
xlabel('v_1')
ylabel('v_2')

colorbar;

figure
hpv4 = htensor({cv, cv});
fxv3 = full(ttt(f,hpv4,[3 4],[1 2]));
[xx1, xx2] = meshgrid(x,x);

contourf(xx1,xx2,max(fxv3,0)','LineColor','none');

name3 = strcat(problem,'_conser_WENO',num2str(order),'_mass_T',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = mass;
save(name3, 'e_elec');

name4 = strcat(problem,'_conser_WENO',num2str(order),'_ener_T',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = [energy(2:end,1), abs(1 - energy(2:end,2)/energy(1,2))];
save(name4, 'e_elec');

name5 = strcat(problem,'_conser_WENO',num2str(order),'_ener_E_T',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = ener_E;
save(name5, 'e_elec');

name6 = strcat(problem,'_conser_WENO',num2str(order),'_mom1_T',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = mom1;
save(name6, 'e_elec');


name7 = strcat(problem,'_conser_WENO',num2str(order),'_mom2_T',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = mom2;
save(name7, 'e_elec');



return







% figure('Renderer', 'painters', 'Position', [10 10 700 400])
% for j = 1:10
%     plot(x,f.U{2}(:,j));
%     hold on
% end



figure('Renderer', 'painters', 'Position', [10 10 500 500])



surf(xx,yy, full(f)','LineStyle','none');

xlim([-pi pi]);

zlim([-0.4, 1.4]);

pbaspect([1 1 1.4]);

M = max(full(f),[],'all')
m = min(full(f),[],'all')





% xlabel('x');
% ylabel('y');
% title(['T=',num2str(T)]);
% title(['max value: ',num2str(M), ', min value: ',num2str(m)]);
% set(gcf,'renderer','opengl');
% saveas(gcf,strcat(problem,'_WENO',num2str(order),'_t',num2str(T),'_',num2str(Nx),'_',num2str(Ny),'_contour_',num2str(opts.max_rank),'.eps'), 'epsc2');
% filename = strcat(problem,'_WENO',num2str(order),'_t',num2str(T),'_',num2str(Nx),'_',num2str(Ny),'_contour_',num2str(opts.max_rank),'.eps');
% %export_fig(filename, '-eps');




figure;
plot((1:Nt-1)'*dt, e_rank);
name1 = strcat(problem,'_conser_WENO',num2str(order),'_rank','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_rank = [(1:Nt-1)'*dt, e_rank];
save(name1, 'e_rank');

name2 = strcat(problem,'_conser_WENO',num2str(order),'_elec','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = [[1:Nt-1]'*dt, e_his];
save(name2, 'e_elec');

name3 = strcat(problem,'_conser_WENO',num2str(order),'_ener','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = [[1:Nt-1]'*dt, ener_his];
save(name3, 'e_elec');

name4 = strcat(problem,'_conser_WENO',num2str(order),'_mass','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'.mat');
e_elec = [[1:Nt-1]'*dt, mass_his];
save(name4, 'e_elec');

disp(name1);

function [f1, F2t, F2] = w_test(f, Nx, Nv, w, Norm, U6, U7)

    % The new idea is to reproject the basis on node (3,4)


    opts.rel_eps = 1e-14;
    opts.max_rank = 1000;


    opts1.rel_eps = 1e-4;
    opts1.max_rank = Nx;



    iw = 1./w;

    vwh = sqrt(w);
    ivwh = sqrt(iw);



    %u1 = rand(Nv,1);
    %u2 = rand(Nv,1);



    ex = ones(Nx,1);
    ev = ones(Nv,1);
    
    vw = w;
    
    ivw = 1./vw;
    
    f_sum = full(ttv(f, {ex,ex,ev,ev}, [1 2 3 4])); 
    
    v = U6(:,2)./vw;
    ee = U6(:,3)./vw;

    %%%
    disp('weighted')

    % here we consider the weighted projection wrt w1 and w2


    norm1 = dot(ev.*vw,ev);
    



    f1 = f;
    f1.U{6} = vwh/norm1*(ev'*f.U{6});
    f1.U{7} = vwh/norm1*(ev'*f.U{7});
    f1 = truncate_std(f1, opts);

    f1 = ttm(f1, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    
    g1 = ttm(f1, {@(y) ivw.*y, @(y) ivw.*y}, [3, 4]);

    f_2 = f - f1; % precompressed with _
    
    f2_bar = ttm(f_2, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    f2bar = truncate_std(f2_bar, opts1);
    
    disp("truncation error for f2bar")
    disp(norm(orthog(f2_bar-f2bar))/norm(f2_bar))    
    
    %postcompressed solution
    f2 = ttm(f2bar, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    g2 = ttm(f2bar, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    
    disp("truncation error for f2")
    disp(norm(f2-f_2)/norm(f_2))  
    
    disp("orthogonal error before truncation:")
    disp(innerprod(g1,f_2))


%     f1_sum = full(ttv(f1, {ex,ex,ev,ev}, [1 2 3 4])); 
%     disp("sum error:")
%     (f_sum - f1_sum)/abs(f_sum)    


    rk = rank(f_2);
%     
%     
    err = 0;
    for i = 1:rk(3)
        B3 = f_2.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron(f_2.U{7}, f_2.U{6}) * B3(:);
    end
    disp('f_2 pre-orthog B3 orthogonal error')
    disp(err)    
    
    
    f2_baro = orthog(f2_bar);
    f_2o = ttm(f2_baro, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    
    
 
    
    rk = rank(f_2o);
%     
%     
    err = 0;
    for i = 1:rk(3)
        B3 = f_2o.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron(f_2o.U{7}, f_2o.U{6}) * B3(:);
    end
    disp('f_2o post-orthog B3 orthogonal error')
    disp(err)
    
    
    

    
    rk = rank(f2);
    err = 0;
    for i = 1:rk(3)
        B3 = f2.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron( f2.U{7}, f2.U{6}) * B3(:);
    end
    
    disp("f2 B3 orthogonal error")
    disp(err)
    
    
    
    % The error should be large as the truncation destories the
    % orthogonality
    
    
    % here we reproject to f2 and get f3 which is able to restore the
    % orthogonality
    
    % We start with f2 (with weights)
    
    mm = 3;
    
    UU6 = [ev v ee];
    UU7 = UU6;
%     Norm = [sum(vw)^2, sum(v.^2.*vw)*sum(vw), sum(v.^2.*vw)*sum(vw), ...
%     2*sum((v.^2-econst).^2.*vw)*sum(vw)];
    
    rk = rank(f2);
    B3 = f2.B{3};
    BB3 = zeros(rk(6)+mm,rk(7)+mm,rk(3));
    BB3(1:rk(6),1:rk(7),:) = B3;

    
    a6 = UU6'* f2.U{6};
    a7 = UU7'* f2.U{7};
    
    B3 = reshape(B3,rk(6)*rk(7),[]);
    b3 = -1/Norm(1)*(kron(a7(1,:),a6(1,:))* B3);
    BB3(rk(6)+1,rk(7)+1,:) = b3;
    
    if mm>1
        b3 = -1/Norm(2)*(kron(a7(1,:),a6(2,:))* B3);
        BB3(rk(6)+2,rk(7)+1,:) = b3;
        b3 = -1/Norm(3)*(kron(a7(2,:),a6(1,:))* B3);
        BB3(rk(6)+1,rk(7)+2,:) = b3;
        if mm>2
            b3 = -1/Norm(4)*((kron(a7(1,:),a6(3,:))+kron(a7(3,:),a6(1,:)))* B3);
            BB3(rk(6)+3,rk(7)+1,:) = b3;
            BB3(rk(6)+1,rk(7)+3,:) = b3;
        end
        
    end
        

    UU = {[],[],[], f2.U{4}, f2.U{5}, [f2.U{6} U6], [f2.U{7} U7]};
    BB = {f2.B{1}, f2.B{2}, BB3, [], [], [], []};
    f3 = htensor(f2.children, f2.dim2ind, UU, BB, false);
    
    
    
    rk = rank(f3);
%     
%     
    err = 0;
    for i = 1:rk(3)
        B3 = f3.B{3}(:,:,i);
        err = err + (kron(ee, ev)+kron(ev,ee))' * kron(f3.U{7}, f3.U{6}) * B3(:);
    end
    
    
    disp('f3 B3 orthogonal error')
    disp(err)
    
    f3_bar = ttm(f3, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    disp("truncation error for f3")
    norm(f3_bar-f2_bar)/norm(f2_bar)  
    
    
    return
    
    
    f2_bar = ttm(f_2, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    f2_bar = truncate_std(f2_bar, opts1);
    f2_t = ttm(f2_bar, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    
    rk = rank(f2_t);
    err = 0;
    for i = 1:rk(3)
        B3 = f2_t.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron( f2_t.U{7}, f2_t.U{6}) * B3(:);
    end
    
    disp("f2 B3 error")
    disp(err)
    
    
    
    f21 = f;
    f21.U{6} = f.U{6} - vw/norm1*(ev'*f.U{6});
    f21.U{7} = vw/norm1*(ev'*f.U{7});    
    
    f22 = f;
    f22.U{7} = f.U{7} - vw/norm1*(ev'*f.U{7});
    
    
    
    f21_bar = ttm(f21, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    f21_bar = truncate_std(f21_bar, opts1);
    f21_t = ttm(f21_bar, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);
    
    rk = rank(f21_t);
    err = 0;
    for i = 1:rk(3)
        B3 = f21_t.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron( f21_t.U{7}, f21_t.U{6}) * B3(:);
    end
    
    disp('f21 error');
    disp(err)
    
    
    f22_bar = ttm(f22, {@(y) ivwh.*y, @(y) ivwh.*y}, [3, 4]);
    f22_bar = truncate_std(f22_bar, opts1);
    f22_t = ttm(f22_bar, {@(y) vwh.*y, @(y) vwh.*y}, [3, 4]);    
    
    
    rk = rank(f22_t);
    err = 0;
    for i = 1:rk(3)
        B3 = f22_t.B{3}(:,:,i);
        err = err + kron(ev, ev)' * kron( f22_t.U{7}, f22_t.U{6}) * B3(:);
    end
    disp('f22 error')
    disp(err)    
    

 











end

function fe =  ttv_energy(f, ev, ee)
    % it works for a special structure of dimension tree
    fe_children = [2, 3; 4, 5; 0, 0; 0, 0; 0, 0];
    fe_dim2ind = [4 5 3];
    
    
    A6 = [ee ev]'*f.U{6};
    A7 = [ev ee]'*f.U{7};
    
    B3 = f.B{3};
    
    r = size(B3);
    r3 = r(3);
    
    U = ones(1,r3);
    
    for i = 1:r3
        U(1,i) = A6(1,:)*B3(:,:,i)*A7(1,:)' + A6(2,:)*B3(:,:,i)*A7(2,:)';
    end
    
    UU = {[],[],U, f.U{4}, f.U{5}};
    BB = {f.B{1}, f.B{2}, [], [], []};
    fe = htensor(fe_children, fe_dim2ind, UU, BB, false);    
    fe = squeeze(fe);
       


end
