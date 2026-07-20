% Define ranges
rankH= 10;
% a ranges from 1e-4 to 0.1
a = linspace(1e-4, 0.1, 100);
% n ranges from 10 to 100
n = linspace(rankH+2, rankH+100, 100);

% Create meshgrid
[A, N] = meshgrid(a, n);

% Calculate y and y1
Y = A.^2 - (A.^2 *rankH./ N);
Y1 = A.^2 + (A.^2 ./ (N-rankH - 1));

% Create a new figure
figure;
hold on;

% Plot surface for Y (Set to Blue)
s1 = surf(A, N, Y);
% Use a specific RGB triplet for blue
set(s1, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'FaceColor', [0 0.4470 0.7410]); 

% Plot surface for Y1 (Set to Red/Orange)
s2 = surf(A, N, Y1);
% Use a specific RGB triplet for orange-red
set(s2, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'FaceColor', [0.8500 0.3250 0.0980]);

% Set Z-axis to log scale
set(gca, 'ZScale', 'log');

% Labels and Title
xlabel('a');
ylabel('n');
zlabel('y (log scale)');
title('Comparison of Surface Functions (Log Scale)');

% Add a legend
legend([s1, s2], {'y = a^2 - a^2/n', 'y1 = a^2 + a^2/(n-1)'}, 'Location', 'best');

% Add grid and rotate view for better visibility
grid on;
view(45, 30);

hold off;