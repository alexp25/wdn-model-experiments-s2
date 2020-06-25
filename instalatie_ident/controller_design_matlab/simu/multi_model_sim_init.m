clc
clear all
close all

error_gain=2;



P1n=[3.7];
P1d=[0.9125 1];

P2n=[4];
P2d=[1.075 1];

P3n=[1.78];
P3d=[0.7875 1];

Pn=[1.8];
Pd=[0.9 1];

Pn_2=[4.2];
Pd_2=[1 1];

% Pn=[3.18];
% Pd=[0.5525 1];


P=tf(P1n,P1d);
[C,info] = pidtune(P,'PI');
% step(feedback(C*P,1))
C1n=[C.Kp C.Ki];
C1d=[1 0];

P=tf(P2n,P2d);
[C,info] = pidtune(P,'PI');
% step(feedback(C*P,1))
C2n=[C.Kp C.Ki];
C2d=[1 0];

P=tf(P3n,P3d);
[C,info] = pidtune(P,'PI');
% step(feedback(C*P,1))
C3n=[C.Kp C.Ki];
C3d=[1 0];