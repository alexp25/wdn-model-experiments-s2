function [data] = avg_outliers(data, threshold)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i=1:length(data)
   if i > 1
       if data(i) - data(i-1) > threshold
        data(i) = data(i-1);
       end
   end
end
end

