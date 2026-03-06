clear
clc
close all
%%
%  This file directory
%  |
%  |-->  patch_generator.m
%  |-->  image_extrator_from_masks_multires.m
%  |-->  cleaned_train_labels.csv
%  |-->  patched_dataset_100/ <--- data_save_directory
%  |     |-->  train_20x/
%  |     |-->  train_10x/
%  |     |-->  train_5x/
%  |_______________

%%

data_save_directory = 'multires_dataset';

base_path     = fileparts(mfilename('fullpath'));
csv_path      = fullfile(base_path, '../cleaned_train_labels.csv');
csv_save_path = fullfile(base_path, data_save_directory, '../cleaned_patched_train_labels.csv');

train20_path  = fullfile(base_path, data_save_directory, 'train_20x');
train10_path  = fullfile(base_path, data_save_directory, 'train_10x');
train5_path   = fullfile(base_path, data_save_directory, 'train_5x');

% crea le cartelle se non esistono
if ~exist(train20_path, 'dir'), mkdir(train20_path); end
if ~exist(train10_path, 'dir'), mkdir(train10_path); end
if ~exist(train5_path,  'dir'), mkdir(train5_path);  end

data = readtable(csv_path);

sample_idx = data.sample_index;
label      = data.label;

label_table = table('Size', [0 3], ...
                    'VariableTypes', {'double', 'string', 'double'}, ...
                    'VariableNames', {'index', 'label', 'original_idx'});
new_sample_idx = 1;

for i = 1 : size(data, 1)
    
    idx = sample_idx{i};

    number_str = extractBetween(idx, "img_", ".png"); 
    img_number = str2double(number_str);
    
    disp(img_number)

    imgpath = fullfile(base_path, '../train_data', sprintf('img_%04d.png', img_number));
    img = imread(imgpath);

    sample_label = label{i};

    [patches, img_padded] = image_extractor_from_masks_multires(img_number, 0);

    for j = 1 : length(patches)

        % update train labels csv
        label_table(end + 1, :) = {new_sample_idx, sample_label, img_number};
        
        % extract original patch
        I = img_padded(ceil(patches{j}(2, 1)) : ceil(patches{j}(2, 2)), ...
                ceil(patches{j}(1, 1)) : ceil(patches{j}(1, 2)), :);

        [h, w, ~] = size(I);

        % ---------- 20x: resize 400x400 -> 100x100 ----------
        I_20x = imresize(I, [100 100]);
        
        
        % ---------- 10x: central crop 200x200 -> resize 100x100 ----------
        crop200 = 200;
        y1_200 = floor((h - crop200)/2) + 1;
        y2_200 = y1_200 + crop200 - 1;
        x1_200 = floor((w - crop200)/2) + 1;
        x2_200 = x1_200 + crop200 - 1;
        I_200  = I(y1_200:y2_200, x1_200:x2_200, :);
        I_10x  = imresize(I_200, [100 100]);

        % ---------- 5x: central crop 100x100 (no resize) ----------
        crop100 = 100;
        y1_100 = floor((h - crop100)/2) + 1;
        y2_100 = y1_100 + crop100 - 1;
        x1_100 = floor((w - crop100)/2) + 1;
        x2_100 = x1_100 + crop100 - 1;
        I_5x   = I(y1_100:y2_100, x1_100:x2_100, :);

        
        filename = sprintf('img_%04d.png', new_sample_idx);

        imwrite(I_20x, fullfile(train20_path, filename));
        imwrite(I_10x, fullfile(train10_path, filename));
        imwrite(I_5x,  fullfile(train5_path,  filename));

       
        new_sample_idx = new_sample_idx + 1;
           



        if (j == 4)
            I_20x_show = I_20x;
            I_10x_show = I_10x;
            I_5x_show = I_5x;
        end

    end

    % Save train csv
    writetable(label_table, csv_save_path);
    

end
