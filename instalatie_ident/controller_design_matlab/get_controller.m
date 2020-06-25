function [ Cn,Cd, C ] = get_controller( Pn,Pd )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
P=tf(Pn,Pd);
% w = 0:1, crossover frequency, increase for response time, default 0.52
[C,info] = pidtune(P,'PI',0.9);
% [C,info] = pidtune(P,'pidf');
% step(feedback(C*P,1))
Cn=[C.Kp C.Ki];
Cd=[1 0];
end

