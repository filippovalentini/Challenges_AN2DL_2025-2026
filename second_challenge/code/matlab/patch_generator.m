clear
clc
close all
%%
%  This file directory
%  |
%  |-->  patch_generator.m
%  |-->  image_extrator_from_masks.m
%  |-->  cleaned_train_labels.csv
%  |-->  data_save_directory/ <--- This must exist before running
%  |     |-->  train_data/
%  |     |     |--> cleaned_patched_train_labels.csv
%  |     |     |--> img_xxxx.png
%  |     |     ...
%  |     |     |___
%  |     |_________
%  |_______________

%% - Path definition

data_save_directory = 'patched_dataset';

base_path = fileparts(mfilename('fullpath'));
csv_path = fullfile(base_path, 'cleaned_train_labels.csv');
csv_save_path = fullfile(base_path, data_save_directory, 'patched_train_labels.csv');
img_save_path = fullfile(base_path, data_save_directory, 'train_data/');

%% - Load data
data = readtable(csv_path);
sample_idx = data.sample_index;
label = data.label;

%% - Create and save patches images

verbose = false;
label_table = table('Size', [0 3], ...
                    'VariableTypes', {'double', 'string', 'double'}, ...
                    'VariableNames', {'index', 'label', 'original_idx'});
new_sample_idx = 1;

for i = 1 : size(data, 1)
    idx = sample_idx{i};
    
    % extract the image number
    number_str = extractBetween(idx, "img_", ".png"); 
    img_number = str2double(number_str);
    
    % define the image path and load image
    imgpath = fullfile(base_path, 'train_data', sprintf('img_%04d.png', img_number));
    img = imread(imgpath);

    % define the label of the image, which will be the label of all the
    % patches generated from that image
    sample_label = label{i};
    
    % extract patches
    patches = image_extractor_from_masks(img_number, verbose);

    % save the patches as new labeled images along with their original WSI
    % index (for Multi Instance Learning purposes)
    for j = 1 : length(patches)
        
        % write patch index, label and original WSI index into csv file
        label_table(end + 1, :) = {new_sample_idx, sample_label, img_number};
        
        % crop WSI image to obtain the patch
        I = img(ceil(patches{j}(2, 1)) : ceil(patches{j}(2, 2)), ceil(patches{j}(1, 1)) : ceil(patches{j}(1, 2)), :);
        
        % save the image
        path = fullfile(img_save_path, sprintf("img_%04d.png", new_sample_idx));
        imwrite(I,path);
        
        % update sample index
        new_sample_idx = new_sample_idx + 1;

    end

    % save csv file
    writetable(label_table, csv_save_path);

end

