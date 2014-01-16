function wid = peak_width_trad( tR, delta_S, delta_H, delta_Cp, col_rad, ...
    coating_thick, plate_num, mod_period)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
T0 = 298.15;
A = (delta_S - delta_Cp .* log(298.15) - delta_Cp)/R;
% check T0 and log(T)
B = (delta_H - delta_Cp * T0)/R;
C = delta_Cp / R;
K = exp(A - B/T + C * log(T));
beta = (col_rad - coating_thick).^2/(2*col_rad*coating_thick);
wid = tR .* (1+K/beta) / sqrt(plate_num);
wid = 4 * sqrt(wid^2 + (mod_period/2)^2);
end
