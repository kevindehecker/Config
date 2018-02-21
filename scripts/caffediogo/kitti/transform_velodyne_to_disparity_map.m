function transform_velodyne_to_disparity_map (im_names_file)
% KITTI RAW DATA DEVELOPMENT KIT
%
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% http://kitti.is.tue.mpg.de/kitti/devkit_raw_data.zip

addpath('E:\TMP\tmptmp\devkit\matlab');

if(nargin < 1)
    im_names_file = 'val_velodyne.txt';
end

graphics = false;

% dev_kit_dir = './';
% addpath(dev_kit_dir);

fid = fopen(im_names_file);
im_names = textscan(fid,'%s','delimiter','\n');
im_names = im_names{1};

% fid = fopen(calib_dir_file);
% calib_dirs = textscan(fid,'%s','delimiter','\n');
% calib_dirs = calib_dirs{1};

cam       = 2; % 0-based index
% load calibration
calib_dir = 'E:/TMP/tmptmp/2011_09_26/'; %calib_dirs{d};
calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));
% compute projection matrix velodyne->image plane
R_cam_to_rect = eye(4);
R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;
n = length(im_names);
parfor frame = 1:n
    vel_file_name = im_names{frame};
    if(exist(vel_file_name, 'file'))
        [filepath,name] = fileparts(vel_file_name);
        
        out_folder = fullfile(filepath, '/../../GT_disp/');
        
        out_im_name = fullfile(out_folder ,[name , '.png']);
%         if(~exist(out_folder, 'dir'))
%             mkdir(out_folder);
%         end
%         
        fprintf('%d / %d\n', frame, n-1);
        
        % load and display image
        %         if(graphics)
        %             img = imread(sprintf('%s/image_%02d/data/%010d.png',base_dir,cam,frame));
        %             fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
        %             imshow(img); hold on;
        %         end
        
        % load velodyne points
        
        fid = fopen(vel_file_name,'rb');
        velo = fread(fid,[4 inf],'single')';
        step = 5; % original script has 5
        velo = velo(1:step:end,:); % remove every 5th point for display speed
        fclose(fid);
        
        % remove all points behind image plane (approximation
        idx = velo(:,1)<5;
        velo(idx,:) = [];
        % project to image plane (exclude luminance)
        velo_img = project(velo(:,1:3),P_velo_to_img);
        
        disps = (64.0*5) ./velo(:,1);
        W = 1242; %size(img, 2);
        H = 375; %size(img, 1);
        new_W = W;
        new_H = H;
        disp_image = construct_disparity_image(velo_img, disps, W, H, new_W, new_H);
        disp_image = uint8(disp_image);
        imwrite(disp_image, out_im_name);
        
        %         if(graphics)
        %             figure();
        %             imagesc(disp_image);
        %             title('disparity image');
        %         end
        
        % write the image in the right place:
        
        
        
    else
        fprintf('File %s does not exist!!!\n', vel_file_name);
    end
    
end
end
