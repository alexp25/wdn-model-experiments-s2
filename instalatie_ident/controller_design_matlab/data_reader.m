clc
clear all
close all



% folder1 = 'sra_nominal_controller_pi'
folder1 = 'sra_matched_controller_pi'
folder1 = 'caracteristica dinamica'
folder1 = 'caracteristica statica pompa 2'
folder_id = 2;

plot_type = 1;

maxy = 700;

folder2_list = {
    'test',
    'noleak',
    'leak_s5',
    'leak_s11'
    };

if folder_id > 0
    folder2 = folder2_list{folder_id}
    M = csvread([folder1 '\' folder2 '\' 'log.csv']);
else
    M = csvread(['log.csv']);
end
% M = csvread('log.csv');

s=size(M);
rows=s(1);
cols=s(2)-1;
N_DATA=3;
t=zeros(rows,cols/N_DATA);
y=t;

for i=1:cols/N_DATA
    t(:,i)=M(:,(i-1)*N_DATA+2)-M(1,(i-1)*N_DATA+2);
    yx = M(:,(i-1)*N_DATA+3);
    for j=1:length(yx)  
        if yx(j) > maxy
            yx(j) = maxy;
        end
    end
    y(:,i)=yx;
end

plot(t,y);
xlabel('time (s)')
ylabel('y (L/h)')
legend('s1','s2','s3','s5','s6','s7','s8','s9','s10','s11')

file_type = 2;
% 1:old format
% 2:new format

fig = figure
t1=t(:,1);
% t1=t1(t1<=140);
r1=rows-length(t1)+1
range=[r1:rows];
% t1=t(range,1);
if file_type==1
    y1=y(range,9);
else
    y1=y(range,13);
    r=y(range,12);
end
t2=t(range,11);
u=y(range,11);

% t3=t2(t2>=1);
% disp('length(t3)')
% length(t3)
% disp('length(u)')
% length(u)
% u_efficiency_test = u(length(u)-length(t3)+1:length(u));
% length(u_efficiency_test)
% % tt = t2(t2>=50);
% % tt(1)
% u_efficiency_test(1)

u_efficiency_test = u;
[command_efficiency_index1, command_efficiency_index2]  = get_command_efficiency_index(u_efficiency_test);

stairs(t1,y1);
if plot_type==1
    subplot(211)
    stairs(t1,u,'r') 
    legend('u (0-255)')
    subplot(212)
    stairs(t1,y1,'b') 
    legend('y (L/h)')
    xlabel('time (s)')
    
    fig = figure
    stairs(t1,y1,'b') 
    legend('y (L/h)')
    xlabel('time (s)')
    ylabel('flow (L/h)')
    
    fig = figure
    plot(u,y1)
    xlabel('u (0-255)')
    ylabel('y (L/h)')
    
    grid on
elseif plot_type==2
    subplot(211)
    hold on
    stairs(t1,r,'g')
    stairs(t1,y1,'b') 
    legend('reference', 'process output')
    xlabel('time (s)')
    ylabel('flow (L/h)')
    grid on
    hold off
    subplot(212)
    stairs(t1,u,'r')
    legend('process input')
    
%     line([t3(1) t3(1)],[0 max(u)]);
    grid on
    xlabel('time (s)')
    ylabel('pump command (PWM)')
elseif plot_type==3
    hold on
    stairs(t1,r,'g')
    stairs(t1,y1,'b') 
    stairs(t1,u,'r') 
    hold off
    legend('r','u (0-255)','y (L/h)')
    xlabel('time (s)')  
end

if folder_id > 0
    saveas(fig,[folder2,'.png'])
else
    saveas(fig,['log.png'])
end







    
    
