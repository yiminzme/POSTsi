%based on v5_tmp_1, remove bias weight term, fixed w_j bugs
%based on v6, fix batch_data bug
%change new variable names
%checking difference on w model
%rough-update
%change input interface
%based on POSText_core_v9_fair_rough
%flexible tolerance threshold
%20190715 add support to binary features

%streaming variational inference for binary tensor data
%prior = {prior_U_m, prior_U_s, prior_lmb_m, prior_lmb_s, prior_w_m, prior_w_s}
%prior_U_m = {d_1 x R, ..., d_K x R}
%prior_U_s = {R x R x d_1, ..., R x R x d_K}
%prior_lmb_m = R x 1
%prior_lmb_s = R x R
%prior_w_m = max(nvec2) x nw2
%prior_w_s = max(nvec2) x nw2
%batch = [batch_bin, batch_cont, batch_y]
%opt = {max_iter, tol}
function post = POSText_core_v10_fair_rough(prior, batch, opt)
    prior_U_m = prior.U_m;
    prior_U_s = prior.U_s;
    prior_lmb_m = prior.lmb_m;
    prior_lmb_s = prior.lmb_s;
    prior_w_m = prior.w_m;
    prior_w_s = prior.w_s;
    nmod = length(prior_U_m);
    nw = size(prior_w_m, 2);
    R = size(prior_U_m{1}, 2);
    [n, nfeat] = size(batch(:, 1:end-1));
    %pre-process batch
    %indices of unique rows in each mode
    uind = cell(nfeat, 1);
    %associated training entries
    data_ind = cell(nfeat, 1);
    for k=1:nfeat
        [uind{k}, ~, ic] = unique(batch(:,k),'stable');
        data_ind{k} = cell(length(uind{k}),1);
        for j=1:length(uind{k})
            data_ind{k}{j} = find(ic == j);
        end
    end
    
    %init variational inference 
    %init post = prior; post stores updated posterior 
    %we can refresh start with the prior, but we can use other initialization
    post_U_m = prior_U_m;
    post_U_s = prior_U_s;
    post_lmb_m = prior_lmb_m; %lambda mean
    post_lmb_s = prior_lmb_s; %lambda variance
    post_w_m = prior_w_m; %{means of weights of f}
    post_w_s = prior_w_s; %{variances of weights of f}
    %randomize mean
    %for k=1:nmod
    %    post_mean{k}(uind{k},:) = randn(length(uind{k}),R);
    %    post_cov{k}(:,:,uind{k}) = reshape(repmat(eye(R), [1, length(uind{k})]), [R,R,length(uind{k})]);
    %end
    y = batch(:,end);
    z = 2*y - 1;
    tol = opt.tol; max_iter = opt.max_iter;
    trials = 0; diff = Inf; iter = 0;
    while diff >= tol && iter < max_iter && diff >= opt.tol0
        iter = iter + 1;
        %reset iter & incrs tol
        if iter == max_iter
            trials = trials + 1;
            tol = opt.tol * opt.tolfactor ^ trials; max_iter = opt.max_iter * (trials+1);
            fprintf("tol = %g\n", tol);
        end
        fprintf('iter = %g, diff = %g\n', iter, diff);
%         fprintf(".");
        
                old_u = cell(nmod,1);
        for k=1:nmod
            old_u{k} = post_U_m{k}(uind{k},:);
        end
        old_lmb = post_lmb_m;
        old_w = post_w_m;
        old_w(isnan(old_w)) = 0;
        for k=1:nmod
            t_nk = ones(n,R);
            t_nk_sq = ones(R,R,n);
            other_modes = setdiff(1:nmod,k);
            for j=1:length(other_modes)
                mod = other_modes(j);
                batch_u_mean = post_U_m{mod}(batch(:,mod),:);
                t_nk = t_nk.*batch_u_mean;
                t_nk_sq = t_nk_sq.*(post_U_s{mod}(:,:,batch(:,mod))...
                    + outer_prod_rows(batch_u_mean));
            end
            lmb_t_nk = t_nk .* post_lmb_m';
            lmb_t_nk_sq = t_nk_sq .* (post_lmb_s + outer_prod_rows(post_lmb_m'));
            %update u
            feat_ind = batch(:,nmod+1:nfeat);
%             wi = cell2mat(cellfun(@(feati)post_w_m(sub2ind(size(post_w_m),feati,1:nw)) ...
%                 ,num2cell(feat_ind,2),'UniformOutput',false));
            wi = cell2mat( cellfun(@(ws, ind) ws(ind), num2cell(post_w_m, 1), ...
                num2cell(feat_ind, 1), 'UniformOutput', false) );
%             featTwi_ = feat.*wi;
%             featTwi_ = wi;
            featTwi_sum = sum(wi,2);
%             featTwi_sum = sum(featTwi_,2)+post_w2m(1,1);
            for j=1:length(uind{k})
                uid = uind{k}(j);
                eid = data_ind{k}{j};
                post_U_s{k}(:, :,uid) = inv(inv(prior_U_s{k}(:,:,uid))...
                    + sum(lmb_t_nk_sq(:,:,eid), 3)); %pay attention here, when batch > 1
                post_U_m{k}(uid,:) = (post_U_s{k}(:,:,uid)...
                    *(prior_U_s{k}(:,:,uid)\(prior_U_m{k}(uid,:).') ...
                    + sum(z(eid).*lmb_t_nk(eid,:) - lmb_t_nk(eid,:).*featTwi_sum(eid), 1)'))'; %pay attention to the sum()
%                     + t_nk(eid,:)'*z(eid).*post_w1m - sum(lmb_t_nk(eid,:).*featTwi_sum(eid))')).';
            end
        end
        t_k = t_nk .* post_U_m{k}(batch(:,k),:);
        t_k_sq = t_nk_sq.*(post_U_s{k}(:,:,batch(:,k))...
                + outer_prod_rows(post_U_m{k}(batch(:,k),:)));
        lmb_t_k = sum(t_k .* post_lmb_m',2);
        feat_val = ones(size(feat_ind));
        feat_val_sq = feat_val.^2;
        %update lambda
        post_lmb_s = inv(inv(prior_lmb_s) + sum(t_k_sq, 3));
        post_lmb_m = post_lmb_s * (prior_lmb_s\prior_lmb_m + t_k'*z - sum(t_k.*featTwi_sum,1)');
        %update w
        for j=1:nw
            ibat = j+nmod;
            eid = data_ind{ibat};
            nj = setdiff(1:nw,j);
            featTwi_nj_sum = sum(wi(:,nj),2);
            post_w_s(uind{ibat},j) = 1./(1./(prior_w_s(uind{ibat},j))...
                + cellfun(@(c)sum(feat_val_sq(c,j),1), eid));
            post_w_m(uind{ibat},j) = ( post_w_s(uind{ibat},j) ...
                .* ( prior_w_s(uind{ibat},j).\prior_w_m(uind{ibat},j) ...
                + cellfun(@(inds)sum(feat_val(inds,j).*(z(inds) - lmb_t_k(inds) ...
                - featTwi_nj_sum(inds)),1), eid) ) );
%                 fprintf("%g ", sum(sum(post_w_m(uind{ibat},j))));
        end
%             fprintf("\n");

        %update z
        z_ = lmb_t_k + featTwi_sum;
        z = z_ + (2*y - 1).*normpdf(z_)./normcdf((2*y-1).*z_);
%             fprintf("%g %g %g\n", z_, z, y);
        diff = 0;
        for k=1:nmod
            diff = diff + sum(sum(abs(old_u{k} - post_U_m{k}(uind{k},:))));
        end
        for k=1:nw
            diff = diff + sum(sum(abs(old_w(uind{k+nmod},k) - post_w_m(uind{k+nmod},k))));
        end
        diff = diff + sum(abs(old_lmb - post_lmb_m));
    end
    fprintf('iter = %g, diff = %g\n', iter, diff);

    post.U_m = post_U_m;
    post.U_s = post_U_s;
    post.lmb_m = post_lmb_m;
    post.lmb_s = post_lmb_s;
    post.w_m = post_w_m;
    post.w_s = post_w_s;
end