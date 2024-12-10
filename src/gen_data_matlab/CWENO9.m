function [dw, jw] = CWENO9(w,dx)
% *************************************************************************
% Input: u(i) = [u(i-2) u(i-1) u(i) u(i+1) u(i+2)];
% Output: res = df/dx;
%
% Based on:
% C.W. Shu's Lectures notes on: 'ENO and WENO schemes for Hyperbolic
% Conservation Laws' 
%
% coded by Manuel Diaz, 02.10.2012, NTU Taiwan.
% edited by Wei Guo 
% *************************************************************************
%
% Domain cells (I{i}) reference:
%
%                |           |   u(i)    |           |
%                |  u(i-1)   |___________|           |
%                |___________|           |   u(i+1)  |
%                |           |           |___________|
%             ...|-----0-----|-----0-----|-----0-----|...
%                |    i-1    |     i     |    i+1    |
%                |-         +|-         +|-         +|
%              i-3/2       i-1/2       i+1/2       i+3/2
%
% ENO stencils (S{r}) reference:
%
%
%                               |___________S2__________|
%                               |                       |
%                       |___________S1__________|       |
%                       |                       |       |
%               |___________S0__________|       |       |
%             ..|---o---|---o---|---o---|---o---|---o---|...
%               | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
%                                      -|
%                                     i+1/2
%
%
%               |___________S0__________|
%               |                       |
%               |       |___________S1__________|
%               |       |                       |
%               |       |       |___________S2__________|
%             ..|---o---|---o---|---o---|---o---|---o---|...
%               | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
%                               |+
%                             i-1/2
%
% WENO stencil: S{i} = [ I{i-2},...,I{i+2} ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: by using circshift over our domain, we are implicitly creating
% favorable code that includes periodical boundary conditions. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lax-Friedrichs Flux Splitting
v=w; u=circshift(w,-1);

%% Right Flux
% Choose the positive fluxes, 'v', to compute the left cell boundary flux:
% $u_{i+1/2}^{-}$
vm4 = circshift(v,4);
vm3  = circshift(v,3);
vm2  = circshift(v,2);
vm1  = circshift(v,1);

vp1  = circshift(v,-1);
vp2 = circshift(v,-2);
vp3 = circshift(v,-3);
vp4 = circshift(v,-4);

% Polynomials
p0n = 1/5*vm4     - 21/20*vm3 + 137/60*vm2 - 163/60*vm1 + 137/60*v;
p1n = -1/20*vm3 + 17/60*vm2 - 43/60*vm1   + 77/60*v         +1/5*vp1;
p2n = 1/30*vm2   - 13/60*vm1 + 47/60*v        + 9/20*vp1       - 1/20*vp2;
p3n = -1/20*vm1 + 9/20*v        + 47/60*vp1     - 13/60*vp2    + 1/30*vp3;
p4n = 1/5*v          + 77/60*vp1  - 43/60*vp2     + 17/60*vp3    - 1/20*vp4;

% p0n = (2*vmm - 7*vm + 11*v)/6;
% p1n = ( -vm  + 5*v  + 2*vp)/6;
% p2n = (2*v   + 5*vp - vpp )/6;

% Smooth Indicators (Beta factors)
% B0n = 13/12*(vmm-2*vm+v  ).^2 + 1/4*(vmm-4*vm+3*v).^2; 
% B1n = 13/12*(vm -2*v +vp ).^2 + 1/4*(vm-vp).^2;
% B2n = 13/12*(v  -2*vp+vpp).^2 + 1/4*(3*v-4*vp+vpp).^2;

B0n = vm4.* (22658 * vm4 - 208501* vm3+364863*vm2 - 288007*vm1 + 86329*v)...
+vm3.*(482963*vm3 -1704396*vm2 + 1358458*vm1 - 411487*v) ...
+vm2.*(1521393*vm2 - 2462076*vm1 +758823*v) ...
+vm1.*(1020563*vm1 - 649501*v)+107918*v.^2;

B1n = vm3.*(6908 * vm3 - 60871 * vm2 + 99213 * vm1 - 70237 * v+18079 * vp1)...
+vm2.*(138563 * vm2 - 464976 * vm1 + 337018 * v - 88297 * vp1) ...
+vm1.*(406293 * vm1 - 611976 * v + 165153 * vp1)...
+v.*(242723 * v - 140251 * vp1) +22658*vp1.^2; 

B2n = vm2.*(6908 * vm2 - 51001 * vm1 + 67923 * v - 38947 * vp1+8209 * vp2)...
+vm1.*(104963 * vm1-299076 * v + 179098 * vp1 - 38947 * vp2)...
+v.*(231153 * v - 299076 * vp1 + 67923 * vp2) ...
+vp1.*(104963 * vp1 - 51001 * vp2) + 6908*vp2.^2;

B3n = vm1.*(22658 * vm1-140251 *  v+165153 *  vp1-88297  * vp2+18079 * vp3)...
+v.*(242723 *  v-611976 * vp1+337018 *  vp2-70237 * vp3)...
+vp1.*(406293 *  vp1-464976  * vp2+99213  * vp3)...
+vp2.*(138563  * vp2-60871 *  vp3)+6908*vp3.^2;

B4n = v.*(107918 * v-649501 * vp1+758823 * vp2 - 411487 * vp3+86329 * vp4)...
+vp1.*(1020563 * vp1-2462076 * vp2+1358458 * vp3-288007 * vp4)...
+vp2.*(1521393 * vp2-1704396 * vp3+364863 * vp4)...
+vp3.*(482963 * vp3-208501 * vp4) +22658*vp4.^2;

% B0n = 13/12*(vmm-2*vm+v  ).^2 + 1/4*(vmm-4*vm+3*v).^2; 
% B1n = 13/12*(vm -2*v +vp ).^2 + 1/4*(vm-vp).^2;
% B2n = 13/12*(v  -2*vp+vpp).^2 + 1/4*(3*v-4*vp+vpp).^2;

% Constants
%d0n = 1/10; d1n = 6/10; d2n = 3/10; epsilon = 1e-6;
d0n = 1/126;  d1n = 10/63; d2n = 10/21; d3n = 20/63; d4n = 5/126; epsilon = 1e-6;
% Alpha weights 

alpha0n = d0n./(epsilon + B0n).^2;
alpha1n = d1n./(epsilon + B1n).^2;
alpha2n = d2n./(epsilon + B2n).^2;
alpha3n = d3n./(epsilon + B3n).^2;
alpha4n = d4n./(epsilon + B4n).^2;

alphasumn = alpha0n + alpha1n + alpha2n + alpha3n + alpha4n;

% ENO stencils weigths
w0n = alpha0n./alphasumn;
w1n = alpha1n./alphasumn;
w2n = alpha2n./alphasumn;
w3n = alpha3n./alphasumn;
w4n = alpha4n./alphasumn;

% Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
hn = w0n.*p0n + w1n.*p1n + w2n.*p2n + w3n.*p3n + w4n.*p4n;

%% Left Flux 
% Choose the negative fluxes, 'u', to compute the left cell boundary flux:
% $u_{i+1/2}^{+}$ 
um4 = circshift(u,4);
um3  = circshift(u,3);
um2  = circshift(u,2);
um1  = circshift(u,1);

up1  = circshift(u,-1);
up2 = circshift(u,-2);
up3 = circshift(u,-3);
up4 = circshift(u,-4);


% Polynomials
% p0p = ( -umm + 5*um + 2*u  )/6;
% p1p = ( 2*um + 5*u  - up   )/6;
% p2p = (11*u  - 7*up + 2*upp)/6;


p4p = 1/5*up4     - 21/20*up3 + 137/60*up2 - 163/60*up1 + 137/60*u;
p3p = -1/20*up3 + 17/60*up2 - 43/60*up1   + 77/60*u         +1/5*um1;
p2p = 1/30*up2   - 13/60*up1 + 47/60*u        + 9/20*um1       - 1/20*um2;
p1p = -1/20*up1 + 9/20*u        + 47/60*um1     - 13/60*um2    + 1/30*um3;
p0p = 1/5*u          + 77/60*um1  - 43/60*um2     + 17/60*um3    - 1/20*um4;


% Smooth Indicators (Beta factors)
% B0p = 13/12*(umm-2*um+u  ).^2 + 1/4*(umm-4*um+3*u).^2; 
% B1p = 13/12*(um -2*u +up ).^2 + 1/4*(um-up).^2;
% B2p = 13/12*(u  -2*up+upp).^2 + 1/4*(3*u -4*up+upp).^2;
B0p = circshift(B0n,-1);
B1p = circshift(B1n,-1);
B2p = circshift(B2n,-1);
B3p = circshift(B3n,-1);
B4p = circshift(B4n,-1);

% Constants
d0p = 5/126; d1p = 20/63; d2p = 10/21; d3p = 10/63; d4p = 1/126; epsilon = 1e-6;

alpha0p = d0p./(epsilon + B0p).^2;
alpha1p = d1p./(epsilon + B1p).^2;
alpha2p = d2p./(epsilon + B2p).^2;
alpha3p = d3p./(epsilon + B3p).^2;
alpha4p = d4p./(epsilon + B4p).^2;

alphasump = alpha0p + alpha1p + alpha2p + alpha3p + alpha4p;

% ENO stencils weigths
w0p = alpha0p./alphasump;
w1p = alpha1p./alphasump;
w2p = alpha2p./alphasump;
w3p = alpha3p./alphasump;
w4p = alpha4p./alphasump;


% Numerical Flux at cell boundary, $u_{i-1/2}^{+}$;
hp = w0p.*p0p + w1p.*p1p + w2p.*p2p + + w3p.*p3p + w4p.*p4p;

%% Compute finite volume residual term, df/dx.
%dw =  (hn - circshift(hn,1))/dx;
dw =  (hn - circshift(hn,1))/dx;
%(hp-circshift(hp,1)+hn-circshift(hn,1))/dx;
jw =  (hp-circshift(hp,1))/dx;
%jw = (hp-circshift(hp,1) - hn + circshift(hn,1))/dx;
%dw = (hn-circshift(hn,1))/dx;

