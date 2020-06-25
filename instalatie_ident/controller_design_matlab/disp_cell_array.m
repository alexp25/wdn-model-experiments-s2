function [displist] = disp_cell_array(Pn,col)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
displist = "";
for i=1:length(Pn)
    Pn1 = Pn{i};
%     disp(Pn1{3})
%     disp(Pn1)
    allOneString = sprintf('%f, ' , Pn1{col});
    allOneString = allOneString(1:end-2);% strip final comma
    displist =  displist + "[" + allOneString + "]" + newline;
end

% disp(displist)
end

