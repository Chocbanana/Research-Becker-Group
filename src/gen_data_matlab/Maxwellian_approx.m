%%
fout = full(f)

Nx = 64;
Nv = 2*Nx

% v = velocity grid (static)
v = -vmax -0.5*hv+ hv*(1:Nv)';
v = v';
% functions of space, diff at different Nt
% Should have properties st 
u = randn(Nx, 1) * hx * 10;
% T0 = static reference eg 0.5
T0 = 0.5;
T = 0.1 * rand(Nx, 1) + T0;


tst = -ones(Nx,1) * v.^2 ./T + 2 * (u./T) * v - u.^2./T * ones(1,Nv);
% exp( - (v - u(x))^2 / T(x) )

% Want the low rank decomp of this
% would pick some divisor to give rank 10 of the exp (or similar)
tst2 = exp(tst / 100);
rank(tst2)
norm(tst2, "fro")
% numel(tst)^3

fnorm = norm(fout, "fro")

mult = f.U{2} * f.B{1} * f.U{3}'
tst3 = norm(fout - mult, "fro")

% TODO: Research how big 1st layer should be in compoarision to dims of
% input


