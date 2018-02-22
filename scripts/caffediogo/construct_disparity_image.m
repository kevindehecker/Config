function D = construct_disparity_image(velo_img, disps, W, H, new_W, new_H)
% function D = construct_disparity_image(velo_img, disps, W, H, new_W, new_H)
% velo_img(1) = x, (2) = y

% k is used in a nearest neighbor sense for the interpolation
k = 1;
max_dist = 1;

% correct for new, probably smaller, image coordinates:
ratio_W = new_W / W;
ratio_H = new_H / H;
velo_img(:,1) = velo_img(:,1) * ratio_W;
velo_img(:,2) = velo_img(:,2) * ratio_H;
n_points = size(velo_img,1);

% % fill the disparity matrix:
% D = zeros(new_H, new_W);
% n_points = size(velo_img,1);
% for x = 1:new_W
%     for y = 1:new_H
%         dists = sqrt(sum((velo_img - repmat([x,y], [n_points, 1])).^2, 2));
%         [min_dist, i] = min(dists);
%         if(min_dist <= max_dist)
%             if(k == 1)
%                 D(y,x) = disps(i);
%             else
%                 % sort the list, etc.
%             end
%         end
%     end
% end
% 
% figure();
% imagesc(D);
% title('Old method');

D = zeros(new_H, new_W);
P = zeros(new_H, new_W);
for p = 1:n_points
    x_int = round(velo_img(p,1));
    y_int = round(velo_img(p,2));
    if(x_int >= 1 && x_int <= new_W && y_int >= 1 && y_int <= new_H)
        D(y_int, x_int) =  D(y_int, x_int) + disps(p);
        P(y_int, x_int) = P(y_int, x_int) + 1;
    end
end

M = P == 0;
P = P + M;
D = D ./ P;

figure();
imagesc(D);
title('New method');

pause;
close all;
