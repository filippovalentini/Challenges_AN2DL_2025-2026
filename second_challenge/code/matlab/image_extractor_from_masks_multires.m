function [patches, img_padded] = image_extractor_from_masks_multires(img_id, verbose)
    base_path = fileparts(mfilename('fullpath'));
    img_path  = fullfile(base_path, '../train_data', sprintf("img_%04d.png", img_id));
    mask_path = fullfile(base_path, '../train_data', sprintf("mask_%04d.png", img_id));
    

    bg_color = uint8([188 188 186]);    
    img  = imread(img_path);
    mask = imread(mask_path); 

    [H, W, C] = size(img);
    
    %% -------- Parameters --------
    % t : half the size of the patch, must reduce t of 0.5 (e.g. t = 111.5 => patch = 224x224)
    t = 111.5;
    
    % outside_ratio : percentage of the patch size that can be outside a
    % mask when random sampling patches inside the region of a BIG mask.
    outside_ratio = 0.5;

    %% - Find connected shapes
    CC    = bwconncomp(mask);

    % Bounding boxes
    props = regionprops(CC, 'BoundingBox', 'Centroid');
    n     = CC.NumObjects;
    
    if verbose
        figure, imshow(mask), title(sprintf('Mask %04d', img_id));
        figure, imshow(img), hold on, title(sprintf('Img %04d', img_id));
    end
    
    p = repmat(struct('mask',[], 'square',[], 'valid',true, ...
                      'square_id',0, 'random_patch', false, ...
                      'random_patches', {}), n, 1);
    
    for i = 1:n
        p(i).valid = true;
    end
    
    %% Patch building

    % Build candidate patches from connected component of the binary mask
    for i = 1:n    
        bb = props(i).BoundingBox;  % [x, y, width, height]
    
        x1 = round(bb(1));
        y1 = round(bb(2));
        x2 = x1 + round(bb(3)) - 1;
        y2 = y1 + round(bb(4)) - 1;

        if verbose
            X = [x1 x2 x2 x1 x1];
            Y = [y1 y1 y2 y2 y1];
            plot(X, Y, 'b-');
            text(x1, y1, sprintf("%d x %d", x2-x1, y2-y1), ...
                 'FontWeight','bold','FontSize',10,'Color','b');
        end
        
        % When mask region is larger than the patch -> random sample some
        % patches (nrand_patches) from the mask region
        if ((x2 - x1) > 2 * t || (y2 - y1) > 2 * t)
            
            x1_sample_region = round([x1 - outside_ratio*2*t, ...
                                      x2 - 2*t*(1 - outside_ratio)]);
            y1_sample_region = round([y1 - outside_ratio*2*t, ...
                                      y2 - 2*t*(1 - outside_ratio)]);
    
            nrand_patches = ceil(((x2-x1) * (y2-y1))/ (2*t*2*t)) + 1;
    
            for k = 1 : nrand_patches
    
                xs = randi([x1_sample_region(1), x1_sample_region(2)]);
                ys = randi([y1_sample_region(1), y1_sample_region(2)]);
    
                x1p = xs; 
                x2p = xs + 2*t; 
                y1p = ys; 
                y2p = ys + 2*t;

                p(i).random_patches{k} = [x1p, x2p; y1p, y2p];
                p(i).random_patch = true;

                if verbose
                    % Draw candidate random patches
                    Xp = [x1p x2p x2p x1p x1p];
                    Yp = [y1p y1p y2p y2p y1p];
                    plot(Xp, Yp, 'g-');
                end
            end
    
        else
            % When mask region is smaller than the patch size, just create
            % one patch centered in the mask region
            l_x = x2 - x1;
            l_y = y2 - y1;
        
            x1_prime = x1 - (t - l_x/2);
            x2_prime = x2 + (t - l_x/2);
        
            y1_prime = y1 - (t - l_y/2);
            y2_prime = y2 + (t - l_y/2);
        
            p(i).mask      = [x1, x2; y1, y2];
            p(i).square    = [x1_prime, x2_prime; y1_prime, y2_prime];
            p(i).square_id = i;
            p(i).random_patch = false;

            if verbose
                Xp = [x1_prime x2_prime x2_prime x1_prime x1_prime];
                Yp = [y1_prime y1_prime y2_prime y2_prime y1_prime];
                plot(Xp, Yp, 'c-');
            end
        end
    end

    % Get the valid p
    p_valid = p([p.valid]);
    
    %% No valid patches
    if isempty(p_valid)
        patches    = {};
        img_padded = img;
        return;
    end

    %% - Patch extension for padding

    % When a patch goes out of the image borders, we pad the image with the
    % background color, because we want the patch to be centered onto the
    % mask region
    min_x1 = +Inf;
    min_y1 = +Inf;
    max_x2 = -Inf;
    max_y2 = -Inf;

    for j = 1:length(p_valid)
        if p_valid(j).random_patch
            for s = 1:length(p_valid(j).random_patches)
                x1 = p_valid(j).random_patches{s}(1,1);
                x2 = p_valid(j).random_patches{s}(1,2);
                y1 = p_valid(j).random_patches{s}(2,1);
                y2 = p_valid(j).random_patches{s}(2,2);

                min_x1 = min(min_x1, x1);
                min_y1 = min(min_y1, y1);
                max_x2 = max(max_x2, x2);
                max_y2 = max(max_y2, y2);
            end
        else
            x1 = p_valid(j).square(1,1);
            x2 = p_valid(j).square(1,2);
            y1 = p_valid(j).square(2,1);
            y2 = p_valid(j).square(2,2);

            min_x1 = min(min_x1, x1);
            min_y1 = min(min_y1, y1);
            max_x2 = max(max_x2, x2);
            max_y2 = max(max_y2, y2);
        end
    end

    min_x1 = floor(min_x1);
    min_y1 = floor(min_y1);
    max_x2 = ceil(max_x2);
    max_y2 = ceil(max_y2);

    %% - Padding
    pad_left   = max(0, 1 - min_x1);
    pad_top    = max(0, 1 - min_y1);
    pad_right  = max(0, max_x2 - W);
    pad_bottom = max(0, max_y2 - H);

    pad_left   = round(pad_left);
    pad_top    = round(pad_top);
    pad_right  = round(pad_right);
    pad_bottom = round(pad_bottom);

    H_new = H + pad_top + pad_bottom;
    W_new = W + pad_left + pad_right;

    %% - Padded image
    img_padded = zeros(H_new, W_new, C, 'like', img);
    for c = 1:C
        img_padded(:,:,c) = bg_color(c);
    end

    img_padded(pad_top+1:pad_top+H, pad_left+1:pad_left+W, :) = img;

    if verbose
        figure, imshow(img_padded), hold on, title('Img padded con patch');
    end

    %% - Patch coordinate translation in the padded image
    patches   = {};
    patch_idx = 1;

    for j = 1:length(p_valid)
        if p_valid(j).random_patch
            for s = 1:length(p_valid(j).random_patches)
                x1 = p_valid(j).random_patches{s}(1,1);
                x2 = p_valid(j).random_patches{s}(1,2);
                y1 = p_valid(j).random_patches{s}(2,1);
                y2 = p_valid(j).random_patches{s}(2,2);

                x1p = x1 + pad_left;
                x2p = x2 + pad_left;
                y1p = y1 + pad_top;
                y2p = y2 + pad_top;

                p_valid(j).random_patches{s} = [x1p, x2p; y1p, y2p];

                patches{patch_idx} = p_valid(j).random_patches{s};
                patch_idx = patch_idx + 1;

                if verbose
                    Xp = [x1p x2p x2p x1p x1p];
                    Yp = [y1p y1p y2p y2p y1p];
                    plot(Xp, Yp, 'c-');
                end
            end
        else
            x1 = p_valid(j).square(1,1);
            x2 = p_valid(j).square(1,2);
            y1 = p_valid(j).square(2,1);
            y2 = p_valid(j).square(2,2);

            x1p = x1 + pad_left;
            x2p = x2 + pad_left;
            y1p = y1 + pad_top;
            y2p = y2 + pad_top;

            p_valid(j).square = [x1p, x2p; y1p, y2p];

            patches{patch_idx} = p_valid(j).square;
            patch_idx = patch_idx + 1;

            if verbose
                Xp = [x1p x2p x2p x1p x1p];
                Yp = [y1p y1p y2p y2p y1p];
                plot(Xp, Yp, 'g-');
            end
        end
    end
end
