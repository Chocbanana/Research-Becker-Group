%low rank explicit method for the heat equation


% Parameters
Lx = pi*2;               % Domain length in the x-direction
Ly = pi*2;               % Domain length in the y-direction
Nx = 10;              % Number of spatial points in the x-direction
Ny = 10;              % Number of spatial points in the y-direction
dx = Lx / Nx;    % Spatial step in the x-direction
dy = Ly / Ny;    % Spatial step in the y-direction
x = linspace(0, Lx, Nx+1); % Spatial grid in the x-direction
y = linspace(0, Ly, Ny+1); % Spatial grid in the y-direction
x = x(1:Nx);
y = y(1:Ny);


% for truncation
opts.max_rank = min(Nx,Ny);
opts.rel_eps = 1e-5;


T = pi;                 % Total simulation time
dt = 0.01;            % Time step
Nt = round(T / dt);    % Number of time steps
alpha = 0.1;          % Thermal diffusivity

% Initial condition (temperature distribution)

u0 = htensor(sin(x), sin(y));
if ~u0.is_orthog()
    u0 = orthog(u0);
end

[X, Y] = meshgrid(x,y);

u_exa = exp(-2*T*alpha)*u0;

% perform reduced SVD on the initial condition
[U, S, V] = svd(u0);
s = diag(S);
threshold = 1e-10;
r = sum(s > threshold);
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

% Finite difference matrices
D2x = (circshift(eye(Nx), [0 1]) - 2 * eye(Nx) + circshift(eye(Nx), [0 -1])) / dx^2;
D2y = (circshift(eye(Ny), [0 1]) - 2 * eye(Ny) + circshift(eye(Ny), [0 -1])) / dy^2;



% Time-stepping using the second-order Runge-Kutta method
u = u0;
for t = 1:Nt
    % First stage
    k1 = alpha * (Ur*Sr*Vr'* D2x + D2y * Ur*Sr*Vr');
    u1 = u + dt * k1; 
    % Second stage
    k2 = alpha * (u1 * D2x + D2y * u1);    
    % Update solution
    u = u + dt * (k1+k2)/2.;
end

% Plot the initial and final temperature distributions
figure;
subplot(1, 3, 1);
surf(X, Y, u0);
title('Initial');
xlabel('x');
ylabel('y');
zlabel('Temperature');

subplot(1, 3, 2);
surf(X, Y, u);
title('Final');
xlabel('x');
ylabel('y');
zlabel('Temperature');

subplot(1, 3, 3);
surf(X, Y, u_exa);
title('Exact');
xlabel('x');
ylabel('y');
zlabel('Temperature');

figure;
surf(X, Y, u-u_exa);
title('error');
xlabel('x');
ylabel('y');
zlabel('Temperature');
