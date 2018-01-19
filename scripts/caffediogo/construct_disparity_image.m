function D = construct_disparity_image(velo_img, disps, W, H)
% function D = construct_disparity_image(velo_img, disps, W, H)
% velo_img(1) = x, (2) = y

% k is used in a nearest neighbor sense for the interpolation
k = 1;
max_dist = 1;

D = zeros(H, W);
n_points = size(velo_img,1);
for x = 1:W
    for y = 1:H
        dists = sqrt(sum((velo_img - repmat([x,y], [n_points, 1])).^2, 2));
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

