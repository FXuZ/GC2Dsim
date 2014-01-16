function center = peak_center( gas_vis, col_len, Pst, flow_rate, ...
    delta_S, delta_H, delta_Cp, col_rad, coating_thick, T, R)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
T0 = 298.15;
A = (delta_S - delta_Cp .* log(298.15) - delta_Cp)/R;
% check T0 and log(T)
B = (delta_H - delta_Cp * T0)/R;
C = delta_Cp / R;
K = exp(A - B/T + C * log(T));
beta = (col_rad - coating_thick).^2/(2*col_rad*coating_thick);
tm = 8/3 * sqrt(pi*col_len^3*gas_vis*T0/(Pst*flow_rate*T));
center = tm .* (1 + K/beta);

end

