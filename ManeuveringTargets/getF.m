%% Define F matrix which is a function of Om (turn rate, possibly time-varying) and T (sample interval, a constant)
function F = getF(Om, T)
if Om % Bar-Shalom, (11.7.1-4)
    F = [1 sin(Om*T)/Om 0 -(1-cos(Om*T))/Om; 0 cos(Om*T) 0 -sin(Om*T); 0 (1-cos(Om*T))/Om 1 sin(Om*T)/Om; 0 sin(Om*T) 0 cos(Om*T)];
else % Om=0, not turning
    F = [1 T 0 0; 0 1 0 0; 0 0 1 T; 0 0 0 1];
end
