%% This demo implements 100 Monte Carlo experiments of Fig. 1 in [2], which averages over 1 000 Monte Carlo experiments.

% For details, see: 
%
% [1] F. K. Teklehaymanot, M. Muma, and A. M. Zoubir, "A Novel 
%     Bayesian Cluster Enumeration Criterion for Unsupervised Learning",
%     IEEE Transactions in signal processing (under review),
%     [Online-Edition: https://arxiv.org/abs/1710.07954v2], 2018.
%
% [2] F. K. Teklehaymanot, M. Muma, and A. M. Zoubir, "Novel Bayesian Cluster
%     Enumeration Criterion for Cluster Analysis With Finite Sample Penalty Term", 
%     IEEE International conference on Acoustics, Speech and Signal Processing (ICASSP) (accepted), 
%     [Online-Edition: https://www.researchgate.net/publication/322918028]


% Copyright (c) 2018 Freweyni K. Teklehaymanot. All rights reserved.


% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published
% by the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.

% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

close all;
clear all;
warning off;


%% User inputs
MC = 100; % number of Monte Carlo experiments
spec_num_clusters = 1:10; % the number of clusters specified by a family of candidate models
num_samples_cluster = 10; % number of data points per cluster

% Initializations
penalty_BIC_NF = cell(1,length(spec_num_clusters)); % initialize the penalty term of BIC_NF (see [2] for details) 
penalty_BIC_N = cell(1,length(spec_num_clusters)); % initialize the penalty term of BIC_N (see [1] for details) 
penalty_BIC_O = cell(1,length(spec_num_clusters)); % initialize the penalty term of BIC_O (see [1] for details) 


for mc = 1:MC
    
    %% Generate data
    [data, num_features, num_samples_total, ~] = generateData(num_samples_cluster, 0);
    
    %% Calculate the penalty term of the Bayesian cluster enumeration criteria
    for l = 1:length(spec_num_clusters)
    
        [est_centroids, est_covmats, est_prob] = modelBasedClustering(data, num_features, num_samples_total, spec_num_clusters(l)); % cluster the data set into the specified number of clusters using the EM algorithm
        [~, est_num_samples_cluster] = hardClusterMemberships(est_prob, spec_num_clusters(l)); % perform hard clustering, see [1] for details
        dup_mat = duplicationMatrix(num_features); % compute the duplication matrix of the covariance matrix

        [~, penalty_BIC_NF{l}(mc), ~] = BIC_NF(num_features, est_num_samples_cluster, est_covmats, dup_mat);
        [~, penalty_BIC_N{l}(mc), ~] = BIC_N(num_features, est_num_samples_cluster, est_covmats);
        [~, penalty_BIC_O{l}(mc), ~] = BIC_O(num_features, num_samples_total, est_num_samples_cluster, est_covmats);

    end
    
end

%% Average the penalty terms over the Monte Carlo experiments
penalty_NF = zeros(1,length(spec_num_clusters));
penalty_N = zeros(1,length(spec_num_clusters));
penalty_O = zeros(1,length(spec_num_clusters));

for l = 1:length(spec_num_clusters)
    
    penalty_NF(l) = sum(penalty_BIC_NF{l})/MC;
    penalty_N(l) = sum(penalty_BIC_N{l})/MC;
    penalty_O(l) = sum(penalty_BIC_O{l})/MC;
    
end

%% PLot results
figure('Name','Penalty term of different Bayesian cluster enumeration criterion') 
plot(spec_num_clusters, -penalty_NF, 'Color', 'r', 'LineWidth', 2)
hold on
plot(spec_num_clusters, -penalty_N, 'Color', 'b', 'Marker', 'diamond', 'LineWidth', 2)
plot(spec_num_clusters, -penalty_O, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2)
xlabel('Number of clusters specified by the candidate models')
ylabel('Penalty term')
legend('BIC_{NF}', 'BIC_N', 'BIC_O', 'Location', 'northwest')
set(gca,'FontSize', 13)
hold off
