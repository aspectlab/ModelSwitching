function F = get_f_x(x, T)
Om=x(5);
F = eye(5);
if abs(Om)>eps % Bar-Shalom, (11.7.2-4)
    F(1:4,2) = [sin(Om*T)/Om cos(Om*T) (1-cos(Om*T))/Om sin(Om*T)]';
    F(1:4,4) = [-(1-cos(Om*T))/Om -sin(Om*T) sin(Om*T)/Om cos(Om*T)]';
    F(1:4,5) = [cos(Om*T)*T*x(2)/Om-sin(Om*T)*x(2)/Om^2-sin(Om*T)*T*x(4)/Om-(-1+cos(Om*T))*x(4)/Om^2
        -sin(Om*T)*T*x(2)-cos(Om*T)*T*x(4)
        sin(Om*T)*T*x(2)/Om-(1-cos(Om*T))*x(2)/Om^2+cos(Om*T)*T*x(4)/Om-sin(Om*T)*x(4)/Om^2
        cos(Om*T)*T*x(2)-sin(Om*T)*T*x(4)];
else % Om=0, Bar-Shalom, (11.7.2-7)
    F(1,2) = T;
    F(3,4) = T;
    F(1:4,5) = [-T^2/2*x(4) -T*x(4) T^2/2*x(2) T*x(2)]';
end
