function [ intens ] = gc2d_dist_mesh( t1, t2, cent1, wid1, cent2, wid2, volatility )
%gc2d_dist_mesh Calculates the simulated intensity distribution of a 2D GC
%on the grid of t1 and t2.
%   Detailed explanation goes here
assert(length(cent1) == length(wid1));
assert(length(cent2) == length(wid2));
assert(length(cent1) == length(cent2));
assert(length(cent1) == length(volatility));
if (length(t1)<=1 && length(t2)<=1)
    intens = 0;
    for i = 1:length(cent1)
        intens = intens + exp(-(t1 - cent1(i))^2/wid1(i)).* ... 
            exp(-(t2 - cent2(i))^2/wid2(i))*volatility(i);
    end
    return;
end
[gt1, gt2] = meshgrid(t1, t2);
intens = zeros(size(gt1));
for i = 1:length(cent1)
    intens = intens + exp(-(gt1 - cent1(i)).^2/wid1(i)).* ... 
        exp(-(gt2 - cent2(i)).^2/wid2(i))*volatility(i);
end

end

