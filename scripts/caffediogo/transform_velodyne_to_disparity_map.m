function transform_velodyne_to_disparity_map (base_dir_file,calib_dir_file)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% http://kitti.is.tue.mpg.de/kitti/devkit_raw_data.zip
addpath('/home/guido/KITTI/devkit/matlab/');

if(nargin < 1)
    base_dir_file = 'base_dirs.txt';
end
if(nargin < 1)
    calib_dir_file = 'calib_dirs.txt';
end

graphics = false;

dev_kit_dir = './';
addpath(dev_kit_dir);

fid = fopen(base_dir_file);
base_dirs = textscan(fid,'%s','delimiter','\n');
base_dirs = base_dirs{1};

fid = fopen(calib_dir_file);
calib_dirs = textscan(fid,'%s','delimiter','\n');
calib_dirs = calib_dirs{1};

cam       = 2; % 0-based index

for d = 1:length(base_dirs)

    fprintf('\nDir %d / %d\n', d, length(base_dirs));
    
    calib_dir = calib_dirs{d};
    base_dir = base_dirs{d};
    
    if(~exist([base_dir '/GT_disp'], 'dir'))
        mkdir([base_dir '/GT_disp']);
    end
    
    % load calibration
    calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
    Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

    % compute projection matrix velodyne->image plane
    R_cam_to_rect = eye(4);
    R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
    P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;

    dir_name = sprintf('%s/image_%02d/data/',base_dir,cam);
    im_names = dir([dir_name '*.png']);

    % frame     = 0; % 0-based index
    
    for frame = 0:length(im_names)-1

        fprintf('%d / %d\n', frame, length(im_names)-1);
        
        % load and display image
        img = imread(sprintf('%s/image_%02d/data/%010d.png',base_dir,cam,frame));
        if(graphics)
            fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
            imshow(img); hold on;
        end

        % load velodyne points
        fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame),'rb');
        velo = fread(fid,[4 inf],'single')';
        step = 1; % original script has 5
        velo = velo(1:step:end,:); % remove every 5th point for display speed
        fclose(fid);

        % remove all points behind image plane (approximation
        idx = velo(:,1)<5;
        velo(idx,:) = [];

        % project to image plane (exclude luminance)
        velo_img = project(velo(:,1:3),P_velo_to_img);

        if(graphics)
            % plot points
            cols = jet;
            for i=1:size(velo_img,1)
                col_idx = round(64*5/velo(i,1)); % obtaining disparities
                plot(velo_img(i,1),velo_img(i,2),'o','LineWidth',4,'MarkerSize',1,'Color',cols(col_idx,:));
            end
        end

        disps = (64.0*5) ./velo(:,1);
        W = size(img, 2);
        H = size(img, 1);
        % output dimensions of the disparity map --> should correspond to
        % output deep neural network!
        new_W = W;
        new_H = H;
        k = 3;
        max_dist = 1;
        disp_image = construct_disparity_image(velo_img, disps, W, H, new_W, new_H);
        
        if(graphics)
            figure();
            imagesc(disp_image);
            title('disparity image');
        end
        
        % write the image in the right place:
        disp_image = uint8(disp_image);
        GT_dir_name = sprintf('%s/GT_disp/', base_dir);
        if(~exist(GT_dir_name, 'dir'))
            mkdir(GT_dir_name);
        end
        im_name = sprintf('%s/GT_disp/%010d.png',base_dir,frame);
        imwrite(disp_image, im_name);
        
    end
end
