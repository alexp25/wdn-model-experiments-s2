clc
clear all
close all

% Pn,Pd: numarator, numitor pentru modelele identificate
% Cn,Cn: numarator, numitor pentru regulatoarele calculate
disp('running controller design')

use_normalized_first_order_model = 0;


Pn={
    {'7-0 (1)', [15.36], [0.903, 1]},    
    {'7-1 (1)', [12.77], [2.37, 1]},  
    {'7-2 (1)', [6.94], [1.4, 1]},    
    {'7-3 (1)', [7.08], [17.28, 1]},
    {'7-4 (1)', [9.24], [6.16, 1]},
    {'7-5 (1)', [7.42], [26.54, 1]},
    {'7-6 (1)', [6.73], [7.62, 1]},
    {'7-7 (1)', [6.01], [4.49, 1]}
    };

disp('showing discrete models')
Ts=0.1

Hp_vect = cell(length(Pn), 1);
Hd_vect = cell(length(Pn), 1);

disp("nums");
disp(disp_cell_array(Pn, 2));
disp("dens");
disp(disp_cell_array(Pn, 3));

% return

    
for i=1:length(Pn)
%     show normalized coefficients (ts+1) for first order model
%     disp(Pn{i})
    Pn1 = Pn{i};
    title1 = Pn1{1};
    pn = Pn1{2};
    pd = Pn1{3};
    
    pn
    pd
    
    if use_normalized_first_order_model
        pdf = pd(length(pd));
        for j=1:length(pd)
            pd(j) = pd(j) / pdf;
        end
        for j=i:length(pn)
            pn(j) = pn(j) / pdf;
        end
    end
    
    Hp = tf(pn, pd);
    
    Hp_vect{i} = Hp;
    
    i
    Hp
    Hpd = c2d(Hp,Ts,'zoh')
    
    [num, den] = tfdata(Hpd, 'v');
    Hd_vect{i} = {title1, num, den};
    step(Hp)
    
    t = "model - " + title1 + " - step response";
    title(t)
    
    if i ~= length(Pn)
        figure
    end
end


C_vect = cell(length(Pn), 1);

Cn = cell(length(Pn),1);
C = cell(length(Pn),1);

displist = cell(length(Pn),1);


for i=1:length(Pn)
    
    Pn1 = Pn{i};
    title1 = Pn1{1};
    pn = Pn1{2};
    pd = Pn1{3};
    
    [Cn{i},Cd{i},C{i}] = get_controller(pn, pd); 
    
    C_vect{i} = tf(Cn{i}, Cd{i});
    
%     disp(['C ',num2str(i)]);
    disp(['C ',num2str(i),' - ', title1, ' - Kp: ',num2str(C{i}.Kp),', Ki: ',num2str(C{i}.Ki),', Kd: ',num2str(C{i}.Kd)]);
    displist{i} = {C{i}.Kp, C{i}.Ki, C{i}.Kd};
%     disp(C{i})
end


disp("nums");
disp(disp_cell_array(Pn, 2));
disp("dens");
disp(disp_cell_array(Pn, 3));


disp("Kps")
disp(disp_cell_array(displist, 1));
disp("Kis")
disp(disp_cell_array(displist, 2));
disp("Kds")
disp(disp_cell_array(displist, 3));


disp("discrete nums")
disp(disp_cell_array(Hd_vect, 2));
disp("discrete dens")
disp(disp_cell_array(Hd_vect, 3));

% evaluate control loops
% return

figure

for i=1:length(Pn)
    Pn1 = Pn{i};
    title1 = Pn1{1};
    pn = Pn1{2};
    pd = Pn1{3};
    
    Hd = Hp_vect{i} * C_vect{i};
    Ho = Hd / (1 + Hd);
    step(Ho);
    
    t = "controller - " + title1 + " - closed loop step response";
    title(t);
    
    if i ~= length(Pn)
        figure
    end
end
    

