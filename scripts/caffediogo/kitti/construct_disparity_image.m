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

% fill the disparity matrix:
D = zeros(new_H, new_W);
n_points = size(velo_img,1);
inds = find(velo_img(:,1) >= 0);
velo_img = velo_img(inds, :);
inds = find(velo_img(:,2) >= 0);
velo_img = velo_img(inds, :);
n_points = size(velo_img,1);

for x = 1:new_W
    for y = 1:new_H
        diff_pos = velo_img - repmat([x,y], [n_points, 1]);
        dists = sum(diff_pos.^2, 2);
        [min_dist, i] = min(dists);
        if(min_dist <= max_dist)
            if(k == 1)
                D(y,x) = disps(i);
            else
                % sort the list, etc.
            end
        end
    end
end

% the other way around:
% P = D;
% for p = 1:n_points
%     int_x = round(velo_img(p,1));
%     int_y = round(velo_img(p,2));
%     if(int_x >= 1 && int_x <= W && int_y >= 1 && int_y <= H)
%         P(int_y, int_x) = P(int_y, int_x) + 1;
%         D(int_y, int_x) = D(int_y, int_x) + disps(p);
%     end
% end
% M = D == 0;
% P = P + M;
% D = D ./ P;

