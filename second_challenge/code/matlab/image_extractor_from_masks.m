function patches = image_extractor_from_masks(img_id, verbose)

    base_path = fileparts(mfilename('fullpath'));
    img_path  = fullfile(base_path, 'train_data', sprintf("img_%04d.png", img_id));
    mask_path = fullfile(base_path, 'train_data', sprintf("mask_%04d.png", img_id));

    img  = imread(img_path);
    mask = imread(mask_path);

    if ~islogical(mask)
        mask_bin = mask > 0;
    else
        mask_bin = mask;
    end

    %% -------- Parameters --------
    % t : half the size of the patch, must reduce t of 0.5 (e.g. t = 111.5 => patch = 224x224)
    t = 111.5;

    % overlapping_factor : percentage of the mask area that must be
    % contained into a square to be associated to that square
    overlapping_factor = 0.7;

    % outside_ratio : percentage of the patch size that can be outside a
    % mask when random sampling patches inside the region of a BIG mask.
    outside_ratio = 0.5;

    %% - Find connected shapes
    CC = bwconncomp(mask_bin);

    % Bounding boxes
    props = regionprops(CC, 'BoundingBox', 'Centroid');

    n = CC.NumObjects;
    if verbose
        figure;
        imshow(img);
        hold on;

        for i = 1:n
            bb = props(i).BoundingBox;  % [x, y, width, height]
            x1 = round(bb(1));
            y1 = round(bb(2));
            x2 = x1 + round(bb(3)) - 1;
            y2 = y1 + round(bb(4)) - 1;


            X = [x1 x2 x2 x1 x1];
            Y = [y1 y1 y2 y2 y1];
            plot(X, Y, 'b-', 'MarkerSize', 8);


        end
    end

    p = repmat(struct('mask',[], 'square',[], 'valid',true, 'square_id',0, ...
                      'random_patch', false, 'random_patches', {{}}), n, 1);

    for i = 1:n
        p(i).valid = true;
    end

    %% - Patch building

    % Build candidate patches from connected component of the binary mask
    for i = 1:n

        bb = props(i).BoundingBox;

        x1 = round(bb(1));
        y1 = round(bb(2));
        x2 = x1 + round(bb(3)) - 1;
        y2 = y1 + round(bb(4)) - 1;
        
        % When mask region is larger than the patch -> random sample some
        % patches (nrand_patches) from the mask region
        if ((x2 - x1) > 2 * t || (y2 - y1) > 2 * t)

            x1_sample_region = round([x1 - outside_ratio*2*t, x2 - 2*t*(1 - outside_ratio)]);
            y1_sample_region = round([y1 - outside_ratio*2*t, y2 - 2*t*(1 - outside_ratio)]);

            nrand_patches = ceil(((x2-x1) * (y2-y1)) / (2*t*2*t)) + 1;

            for k = 1:nrand_patches
                xs = randi([x1_sample_region(1), x1_sample_region(2)]);
                ys = randi([y1_sample_region(1), y1_sample_region(2)]);

                x1r = xs; x2r = xs + 2*t;
                y1r = ys; y2r = ys + 2*t;

                p(i).random_patches{k} = [x1r, x2r; y1r, y2r];
                p(i).random_patch = true;

                if verbose
                    % Draw candidate random patches
                    X = [x1r x2r x2r x1r x1r];
                    Y = [y1r y1r y2r y2r y1r];
                    plot(X, Y, 'g-', 'MarkerSize', 6);
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
                plot(Xp, Yp, 'c-', 'MarkerSize', 8);
            end

        end
    end

    %% - Check for overlapping squares and reduce them accordingly
    idx_regular = find(~[p.random_patch]);

    for ii = 1:length(idx_regular)
        i = idx_regular(ii);
        x1_sq = p(i).square(1,1); x2_sq = p(i).square(1,2);
        y1_sq = p(i).square(2,1); y2_sq = p(i).square(2,2);

        for jj = 1:length(idx_regular)
            j = idx_regular(jj);
            if i ~= j
                x1_mask = p(j).mask(1,1); x2_mask = p(j).mask(1,2);
                y1_mask = p(j).mask(2,1); y2_mask = p(j).mask(2,2);
                
                % Reduce the number of patches by eliminating the squares
                % assigned to regions of the mask that are fully contained in another
                % square
                if isMaskInsideSquare(x1_sq, x2_sq, y1_sq, y2_sq, ...
                                      x1_mask, x2_mask, y1_mask, y2_mask, ...
                                      overlapping_factor) && p(i).valid
                    p(j).valid = false;
                    p(j).square_id = i;
                end
            end
        end
    end

    % Get valid patches
    p_valid = p([p.valid]);

    %% Plot the mask overlay
    if verbose
        figure;
        imshow(img);
        hold on;
    
        % Overlay binary mask in transparent GREEN
        green = cat(3, zeros(size(mask_bin)), ones(size(mask_bin)), zeros(size(mask_bin)));
        hMask = imshow(green);
        set(hMask, 'AlphaData', 0.25 * double(mask_bin));
    end

    % Collect final patch squares into "patches"
    patches = {};
    patch_idx = 1;

    for j = 1:length(p_valid)
        if p_valid(j).random_patch
            for s = 1:length(p_valid(j).random_patches)
                x1p = p_valid(j).random_patches{s}(1,1);
                x2p = p_valid(j).random_patches{s}(1,2);
                y1p = p_valid(j).random_patches{s}(2,1);
                y2p = p_valid(j).random_patches{s}(2,2);
                
                % Check if the patch ends up outside of the image borders
                % and move it inside
                [x1p, x2p, y1p, y2p] = clampSquareToImage(x1p, x2p, y1p, y2p, size(img,2), size(img,1));

                p_valid(j).random_patches{s}(1,1) = x1p;
                p_valid(j).random_patches{s}(1,2) = x2p;
                p_valid(j).random_patches{s}(2,1) = y1p;
                p_valid(j).random_patches{s}(2,2) = y2p;

                patches{patch_idx} = p_valid(j).random_patches{s};
                patch_idx = patch_idx + 1;


                drawBoxWithMarkers(x1p, x2p, y1p, y2p, 7);
            end
        else
            x1p = p_valid(j).square(1,1);
            x2p = p_valid(j).square(1,2);
            y1p = p_valid(j).square(2,1);
            y2p = p_valid(j).square(2,2);

            [x1p, x2p, y1p, y2p] = clampSquareToImage(x1p, x2p, y1p, y2p, size(img,2), size(img,1));

            p_valid(j).square(1,1) = x1p;
            p_valid(j).square(1,2) = x2p;
            p_valid(j).square(2,1) = y1p;
            p_valid(j).square(2,2) = y2p;

            patches{patch_idx} = p_valid(j).square;
            patch_idx = patch_idx + 1;
            
            if verbose
                drawBoxWithMarkers(x1p, x2p, y1p, y2p, 7);
            end
        end
    end

    %% -------------------- Helper functions --------------------
    
    function contained = isMaskInsideSquare(x1_sq, x2_sq, y1_sq, y2_sq, ...
                                            x1_mask, x2_mask, y1_mask, y2_mask, ...
                                            overlapping_factor)
        x1_sq0 = min(x1_sq, x2_sq);  x2_sq0 = max(x1_sq, x2_sq);
        y1_sq0 = min(y1_sq, y2_sq);  y2_sq0 = max(y1_sq, y2_sq);
        x1_m0  = min(x1_mask, x2_mask); x2_m0 = max(x1_mask, x2_mask);
        y1_m0  = min(y1_mask, y2_mask); y2_m0 = max(y1_mask, y2_mask);

        sq_poly   = polyshape([x1_sq0 x2_sq0 x2_sq0 x1_sq0], [y1_sq0 y1_sq0 y2_sq0 y2_sq0]);
        mask_poly = polyshape([x1_m0  x2_m0  x2_m0  x1_m0 ], [y1_m0  y1_m0  y2_m0  y2_m0 ]);

        inter_poly = intersect(sq_poly, mask_poly);
        inter_area = area(inter_poly);
        mask_area  = area(mask_poly);

        if mask_area == 0
            contained = false;
            return;
        end

        contained = (inter_area / mask_area) >= overlapping_factor;
    end

    function [x1_new, x2_new, y1_new, y2_new] = clampSquareToImage(x1, x2, y1, y2, W, H)
        x1a = min(x1, x2); x2a = max(x1, x2);
        y1a = min(y1, y2); y2a = max(y1, y2);

        shift_x = 0;
        if x1a < 1
            shift_x = 1 - x1a;
        elseif x2a > W
            shift_x = W - x2a;
        end

        shift_y = 0;
        if y1a < 1
            shift_y = 1 - y1a;
        elseif y2a > H
            shift_y = H - y2a;
        end

        x1_new = x1a + shift_x;
        x2_new = x2a + shift_x;
        y1_new = y1a + shift_y;
        y2_new = y2a + shift_y;
    end

    function drawBoxWithMarkers(x1, x2, y1, y2, msz)
       
        step = 1; 
        xs_top = x1:step:x2;  ys_top = y1 * ones(size(xs_top));
        xs_bot = x1:step:x2;  ys_bot = y2 * ones(size(xs_bot));

        ys_left  = y1:step:y2; xs_left  = x1 * ones(size(ys_left));
        ys_right = y1:step:y2; xs_right = x2 * ones(size(ys_right));

        plot(xs_top,  ys_top,  'k.', 'MarkerSize', msz);
        plot(xs_bot,  ys_bot,  'k.', 'MarkerSize', msz);
        plot(xs_left, ys_left, 'k.', 'MarkerSize', msz);
        plot(xs_right,ys_right,'k.', 'MarkerSize', msz);
    end
end
