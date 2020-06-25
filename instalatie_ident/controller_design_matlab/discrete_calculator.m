clc
clear all

num=[3.7]
den=[0.91 1]
Hp = tf(num,den)
Ts=0.1
Hpd = c2d(Hp,Ts,'zoh')