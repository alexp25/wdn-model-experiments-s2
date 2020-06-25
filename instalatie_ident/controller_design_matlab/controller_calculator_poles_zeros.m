clc
clear all
close all

% Pn,Pd: numarator, numitor pentru modelele identificate
% Cn,Cn: numarator, numitor pentru regulatoarele calculate
disp('running controller design')
% Pn={[0.98],
%     [0.74],
%     [0.44]
%     };

Pn={[2],
    [1],
    [1]
    };

% Pn={[1.9],
%     [2.5],
%     [1],
%     [1],
%     [0],
%     [2.5]
%     };

Pd={[0.5,1],
    [0.425,1],
    [0.4,1]
    };

K0 = 1;
T0 = 2.5;
H0 = tf([K0],[T0 1]);

% zeta=0.7
% wn=1
% H0 = tf([wn^2],[1 2*zeta*wn wn^2])
Hd = H0/(1-H0);

step(H0)

disp('showing discrete models')
Ts=0.1

C = cell(length(Pn),1);

for i=1:length(Pn)
    Hp = tf(Pn{i},Pd{i});
    disp(i)
    Hpd = c2d(Hp,Ts,'zoh')
%     step(Hp)
%     figure
    Hr = Hd/Hp;
    C{i} = minreal(Hr);
    C{i}
end





