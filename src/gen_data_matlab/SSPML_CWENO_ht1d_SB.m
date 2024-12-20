% VP 1d

% SRB: setup path
pd='/Users/srbecker/Library/CloudStorage/OneDrive-UCB-O365/Documents/MATLAB/CHaRMNET/Bhavana/JingmeiDropbox1D1VVlasov tensor for vp';
addpath( fullfile(pd,'htucker_1.2') )

clear all
a = []; % coefficient
% prob = 2; % problem
prob = 6;

order = 5;
% T = 40;

T = 1;

Nx = 128;
Nv = 2*Nx;

% for truncation
opts.max_rank = min(Nx,Nv); % no effective constraint
opts.rel_eps = 1e-5; % if 1e-16 then no truncation. Default was 1e-5
% opts.rel_eps = 1e-16;
opts.rel_eps = 1e-3;
% and full(f)

% NO_TENSOR = ( opts.rel_eps < 1e-15 );
NO_TENSOR = false;

% and learn low-rank of exp( .. )

% Compare to CUR/ID/etc.

problem = [];
if prob==1
    problem = 'weak1d';  % default
    a = 0.015;
elseif prob ==2
    problem = 'strong1d';
    a = 0.5;
elseif prob == 3
    problem = 'twostream1d';
elseif prob == 4
    problem = 'twostream1dii';
    a = 0.01;
elseif prob == 5
    problem = 'twostream1diii';
    a = 0.05;
elseif prob == 6
    problem = 'random_initial_condition';
    a = 0; % not used
end


if order == 5
    CWENO = @CWENO5;
elseif order ==9
    CWENO = @CWENO9;
else
    error("wrong spatial order");
end




kt = 2; % temporal order


kx = 0.5; % wavenumber in x
vmax = 2*pi;
if prob ==3
    kx = 0.2;
    vmax = 9;
end
Lx = 2*pi/kx;
Lv = vmax * 2; % [-6,6]

kv = 2*pi/Lv;

hx = Lx/Nx;
hv = Lv/Nv;

dt = min(hx,hv)/20;

if prob == 6
    T = 10*dt;
    disp('Warning, only running simulation for 10 time steps')
end



Nt = floor(T/dt);


%  Solution will be Nt x Nx x Nv, stored as 8 bytes
GB = 1024^3;
storageCost = Nt * Nx * Nv * 8 / GB
fprintf('Predicted storage is %.1f GB\n', storageCost )


% for computing E via fft
Nk = [0:Nx/2-1 0 -Nx/2+1:-1]';
Nk(1) = 1;
Nk(Nx/2+1) = 1;
Nk = Nk*1i*kx;


Nkx = [0:Nx/2-1 0 -Nx/2+1:-1]'; % for first derivative
Nnx = abs(Nkx)*2/Nx;





x = hx*(1:Nx)';
v = -vmax -0.5*hv+ hv*(1:Nv)';


% for computing rho via contraction

htrho =  htensor({ones(Nv,1)});

htj =  htensor({v});

hte =  htensor({v.^2});


%% landau damping
if prob == 1 || prob==2
    f = htensor({1/sqrt(2*pi)*(1+ a*cos(kx*x)), exp(-v.^2/2)}); % create a htd tensor from CP format
elseif prob ==3
    a1 = 1.; b1 = 2.4; eps1 = 1e-3; f = htensor({1/(2*sqrt(2*pi))*(1+ eps1*cos(kx*x)), exp(-((v-b1)/(a1*sqrt(2))).^2) + exp(-((v+b1)/(a1*sqrt(2))).^2)});
elseif prob ==4
    f = htensor({2/(7*sqrt(2*pi))*(1+ a*((cos(2*kx*x) + cos(3*kx*x))/1.2 + cos(kx*x))), (1+5*v.^2).*exp(-v.^2/2)}); % create a htd tensor from CP format
    %f = htensor({2/(7*sqrt(2*pi))*(1+ a*((cos(2*kx*(x+0.5*Lx)) + cos(3*kx*(x+0.5*Lx)))/1.2 + cos(kx*(x+0.5*Lx)))), (1+5*v.^2).*exp(-v.^2/2)}); % create a htd tensor from CP format
elseif prob ==5
    f = htensor({1/sqrt(2*pi)*(1+ a*cos(kx*x)), v.^2.*exp(-v.^2/2)}); % create a htd tensor from CP format
elseif prob == 6
    y_x = smooth( x,randn(size(x)),0.75, 'loess');
    y_v = smooth( v,randn(size(v)),0.75, 'loess');
    f = htensor({y_x, y_v}); % create a htd tensor from CP format
end

%%

if ~f.is_orthog()
    f = orthog(f);
end

tmass = 1;
if prob == 4
    tmass = 12/7;
end
    
DO_FAST = true;
DO_FAST_CHECK = false;

% figure;
% he = animatedline;

[xx, vv] = meshgrid(x,v);

e_his = [];
e_rank = [];
mass_his = [];
ener_his = [];



tic;

list{1} = f;

%list(1) = f;

for i=2:3
    
    % rk2 stage 1
    
    ft = f; % f is Nx x Nv  (full(f) will be matrix)
    
    if NO_TENSOR
        % Rho is Nx x 1,  htrho is Nv x 1
        rho = sum( full(f), 1 );
    else
        rho = ttt(f, htrho, 2, 1);  % tensor contraction; htensor({ones(Nv,1)});
    end

    
    
    rhof = hv*full(rho) - tmass;
    
    
    mass = sum(rhof)*hx;
    
    
    % Definition: hte =  htensor({v.^2}); size Nv x 1
    if NO_TENSOR
        Kinetic = f * (v.^2);
    else
        Kinetic = ttt(f, hte, 2, 1);
    end

    k_ener = sum(full(Kinetic))*hv*hx; % kinetic energy
    
    

    
    
    r_hat = fft(rhof);
    
    h_hat = r_hat./Nk;
    h_hat(1) = 0;
    h_hat(Nx/2+1) = 0;
    
    E = real(ifft(h_hat));
    
    
    
    
    e_ener = dot(E,E)*hx; % electric energy
        
        
    if(i==2)
        ener_int = k_ener + e_ener;
    end
        
    
    ener = (k_ener + e_ener - ener_int)/ener_int;
    
    
    e_his = [e_his; e_ener];
    
    mass_his = [mass_his; mass];
    
    ener_his = [ener_his; ener];

    
    
    Ux = f.U{2};  %these are the factorizations
    Uv = f.U{3};  % so full(f) = Ux * f.B{1} *  Uv'
    
    DUxm = zeros(size(Ux));
    DUvm = zeros(size(Uv));
    DUxp = zeros(size(Ux));
    DUvp = zeros(size(Uv));
    
    % Apply differentiation...
    for j=1:size(Ux,2)
        
        [du, ju] = CWENO(Ux(:,j),hx);
        DUxm(:,j) = du;% positive velocity
        DUxp(:,j) = ju;
        
    end
    
    %     beta = max(abs(E));
    
    for j=1:size(Uv,2)
        
        [du, ju] = CWENO(Uv(:,j),hv);
        DUvm(:,j) = du;% positive velocity
        DUvp(:,j) = ju;
   
    end
    
    fxp = f;
    fvp = f;
    fxp.U{2} = DUxp;
    fvp.U{3} = DUvp;
    
    fxm = f;
    fvm = f;
    fxm.U{2} = DUxm;
    fvm.U{3} = DUvm;
    
    % Upwind
    f1x1 = - dt * (ttm(fxm, @(y) max(v,0).*y, 2) + ttm(fxp, @(y) min(v,0).*y, 2)); % v fx
    f1v1 = - dt * (ttm(fvm, @(y) max(E,0).*y, 1) + ttm(fvp, @(y) min(E,0).*y, 1)); % E fv
    
    fsum = {f, f1x1, f1v1};
    
    if DO_FAST
        if DO_FAST_CHECK
            full_f = full(htensor.truncate_sum(fsum, opts));

            f = matrix_truncate_sum(opts,f, f1x1, f1v1);
            fprintf('Err: %.2e\n', norm( full_f - full(f), 'fro' ) )
        else
            f = matrix_truncate_sum(opts,f, f1x1, f1v1);
        end
    else
        f = htensor.truncate_sum(fsum, opts);
    end
    
    % rk stage 2
    
    rho = ttt(f, htrho, 2, 1);
    
    
    rhof = hv*full(rho) - tmass;
    
    
    
    r_hat = fft(rhof);
    
    h_hat = r_hat./Nk;
    h_hat(1) = 0;
    h_hat(Nx/2+1) = 0;
    
    E = real(ifft(h_hat));
    
    
    
    Ux = f.U{2};
    Uv = f.U{3};
    
    DUxm = zeros(size(Ux));
    DUvm = zeros(size(Uv));
    DUxp = zeros(size(Ux));
    DUvp = zeros(size(Uv));
    
    for j=1:size(Ux,2)
        
        [du, ju] = CWENO(Ux(:,j),hx);
        DUxm(:,j) = du;% positive velocity
        DUxp(:,j) = ju;
        
   
    end
    
   
    for j=1:size(Uv,2)
       
        [du, ju] = CWENO(Uv(:,j),hv);
        DUvm(:,j) = du;% positive velocity
        DUvp(:,j) = ju;
      
    end
    
    fxp = f;
    fvp = f;
    fxp.U{2} = DUxp;
    fvp.U{3} = DUvp;
    
    fxm = f;
    fvm = f;
    fxm.U{2} = DUxm;
    fvm.U{3} = DUvm;
    
    
    f1x1 = - 0.5*dt * (ttm(fxm, @(y) max(v,0).*y, 2) + ttm(fxp, @(y) min(v,0).*y, 2)); % v fx
    f1v1 = - 0.5*dt * (ttm(fvm, @(y) max(E,0).*y, 1) + ttm(fvp, @(y) min(E,0).*y, 1)); % E fv
    
    fsum = {0.5*ft, 0.5*f, f1x1, f1v1};
    

%     f = htensor.truncate_sum(fsum, opts);
    if DO_FAST
        if DO_FAST_CHECK
            full_f = full(htensor.truncate_sum(fsum, opts));

            f = matrix_truncate_sum(opts,.5*f, f1x1, f1v1, .5*ft ); % SRB trying
            fprintf('Err: %.2e\n', norm( full_f - full(f), 'fro' ) )
        else
            f = matrix_truncate_sum(opts,.5*f, f1x1, f1v1, .5*ft ); % SRB trying
        end
    else
        f = htensor.truncate_sum(fsum, opts);
    end
    
    
    list{i} = f;
    
    rk = rank(f);

    
    

    
    e_rank = [e_rank; rk(2)];
    
%     addpoints(he, i*dt, log(dot(E,E)*hx));
%     drawnow
    
    %rk = rank(f);
    %disp([i*dt, rk]);
    
%     contourf(xx,vv,full(f)');
%     title(['t=', num2str(i*dt), ' rank=',num2str(rk)]);
%     drawnow
%     
    
    
    
end

% return;

% fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
% disp('Time  Rnk 1  Rnk 2  Rnk 3')
% fprintf('%6.4f %3d %3d %3d\n', i*dt, rk(1),rk(2), rk(3) )
disp('        ')
for i = 4:Nt
    %f = list{3};
    
    rho = ttt(f, htrho, 2, 1);
    
    
    rhof = hv*full(rho) - tmass;
    
    mass = sum(rhof)*hx;
    
    
    Kinetic = ttt(f, hte, 2, 1);

    k_ener = sum(full(Kinetic))*hv*hx; % kinetic energy
    
    

    
    
    r_hat = fft(rhof);
    
    h_hat = r_hat./Nk;
    h_hat(1) = 0;
    h_hat(Nx/2+1) = 0;
    
    E = real(ifft(h_hat));
    
    
    Ux = f.U{2};
    Uv = f.U{3};
    
    DUxm = zeros(size(Ux));
    DUvm = zeros(size(Uv));
    DUxp = zeros(size(Ux));
    DUvp = zeros(size(Uv));
    
    for j=1:size(Ux,2)
        
        [du, ju] = CWENO(Ux(:,j),hx);
        DUxm(:,j) = du;% positive velocity
        DUxp(:,j) = ju;
        
    end
    
    %     beta = max(abs(E));
    
    for j=1:size(Uv,2)
        
        [du, ju] = CWENO(Uv(:,j),hv);
        DUvm(:,j) = du;% positive velocity
        DUvp(:,j) = ju;
   
    end
    
    fxp = f;
    fvp = f;
    fxp.U{2} = DUxp;
    fvp.U{3} = DUvp;
    
    fxm = f;
    fvm = f;
    fxm.U{2} = DUxm;
    fvm.U{3} = DUvm;
    
    
    f1x1 = - 1.5*dt * (ttm(fxm, @(y) max(v,0).*y, 2) + ttm(fxp, @(y) min(v,0).*y, 2)); % v fx
    f1v1 = - 1.5*dt * (ttm(fvm, @(y) max(E,0).*y, 1) + ttm(fvp, @(y) min(E,0).*y, 1)); % E fv
    
    
    % u^{n+1} = 0.75*(u^n+2dt*F(u^n)) + 0.25*u^{n-2}
    fsum = {0.25*list{1}, 0.75*f, f1x1, f1v1};

    % Todo: scalars separate
%     full_f = matrix_truncate_sum(opts, 0.25*list{1}, f1x1, f1v1, 0.75*f); % SRB trying

    if DO_FAST
        if DO_FAST_CHECK
            full_f = full(htensor.truncate_sum(fsum, opts));

            f = matrix_truncate_sum(opts, 0.25*list{1}, f1x1, f1v1, 0.75*f); % SRB trying
            fprintf('Err: %.2e\n', norm( full_f - full(f), 'fro' ) )
            if norm( full_f - full(f), 'fro' ) > 1e-5
                disp('Found bad error!')
                keyboad
            end
        else
            f = matrix_truncate_sum(opts, 0.25*list{1}, f1x1, f1v1, 0.75*f); % SRB trying
        end
    else
        f = htensor.truncate_sum(fsum, opts);
    end

%     f = htensor.truncate_sum(fsum, opts);
    
    list{1} = list{2};
    list{2} = list{3};
    list{3} = f;
    
    rk = rank(f);
%     
%     contourf(xx,vv,full(f)');
%     title(['t=', num2str(i*dt)]);
%     drawnow

%     disp([i*dt,rk]);
%     fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
%     fprintf('%6.4f %3d %3d %3d\n', i*dt, rk(1),rk(2), rk(3) )
    if ~DO_FAST_CHECK
        fprintf('\b\b\b\b\b\b%5.2f%%',i/Nt*100)
    end
    
    
    e_ener = dot(E,E)*hx; % electric energy
    
    ener = (k_ener + e_ener - ener_int)/ener_int;
    
    
    e_his = [e_his; e_ener];
    
    mass_his = [mass_his; mass];
    
    ener_his = [ener_his; ener];

    
    e_rank = [e_rank; rk(2)];
    
    
    
    
    
    
end


toc;

disp(toc);
%%
[xx, vv] = meshgrid(x,v);

figure;

%colormap(jet);



if prob==2
    cont = 0:0.01:0.5;
    zlim = [0, 0.5];
elseif prob==4
    cont = 0:0.01:0.5;
    zlim = [0, 0.5];
elseif prob ==5
    cont = 0:0.01:0.32;
end

[C, H] = contourf(xx,vv, max(0,full(f)'),cont);

set(H,'LineColor','none');

colorbar;
%caxis(zlim);

xlabel('x');
ylabel('v');
title(['t=',num2str(T)]);
set(gcf,'renderer','opengl');
saveas(gcf,strcat(problem,'_WENO_t',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_contour_',num2str(opts.max_rank),'.eps'), 'epsc2');
filename = strcat(problem,'_WENO_t',num2str(T),'_',num2str(Nx),'_',num2str(Nv),'_contour_',num2str(opts.max_rank),'.eps');
%export_fig(filename, '-eps');

figure;

semilogy([1:Nt-1]*dt, e_his);


figure;
plot((1:Nt-1)'*dt, e_rank);
name1 = strcat(problem,'_WENO',num2str(order),'_rank','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'_T',num2str(floor(T)),'.mat');
e_rank = [[1:Nt-1]'*dt, e_rank];
save(name1, 'e_rank');

name2 = strcat(problem,'_WENO',num2str(order),'_elec','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'_T',num2str(floor(T)),'.mat');
e_elec = [[1:Nt-1]'*dt, e_his];
save(name2, 'e_elec');

name3 = strcat(problem,'_WENO',num2str(order),'_ener','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'_T',num2str(floor(T)),'.mat');
e_elec = [[1:Nt-1]'*dt, ener_his];
save(name3, 'e_elec');

name4 = strcat(problem,'_WENO',num2str(order),'_mass','_',num2str(Nx),'_',num2str(Nv),'_',num2str(opts.rel_eps),'_',num2str(opts.max_rank),'_T',num2str(floor(T)),'.mat');
e_elec = [[1:Nt-1]'*dt, mass_his];
save(name4, 'e_elec');

disp(name1);

