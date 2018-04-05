%% The main file that runs Bayesian cluster enumeration algorithms

% For details, see: 
%
% [1] F. K. Teklehaymanot, M. Muma, and A. M. Zoubir, "A Novel
%     Bayesian Cluster Enumeration Criterion for Unsupervised Learning",
%     IEEE Transactions in signal processing (under review),
%     [Online-Edition: https://arxiv.org/abs/1710.07954v2], 2018.
%
% [2] F. K. Teklehaymanot, M. Muma, and A. M. Zoubir, "Novel Bayesian
%     Cluster Enumeration Criterion for Cluster Analysis With Finite Sample Penalty Term", 
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


% Outputs:
%         est_num_clusters: estimated number of clusters in the observed data set
%         est_cluster_centroids: estimated cluster centroids
%         est_cluster_covmats: estimated cluster covariance matrices


close all;
clear all;
warning off;


%% User inputs
spec_num_clusters = 1:10; % the number of clusters specified by a family of candidate models
num_samples_cluster = 10; % number of samples per cluster
criterion = 'BIC_NF'; % type of cluster enumeration criterion % possible inputs are 'BIC_N', 'BIC_NF', or 'BIC_O' (See [1] and [2] for details on the criteria)
plotting = 1; % if set to 1 it plots the raw data and the calculated Bayesian Information Criterion (BIC) as a function of the specified number of clusters by the candidate models 

%% Generate data
[data, num_features, num_samples_total, num_clusters] = generateData(num_samples_cluster, plotting);

%% Perform Bayesian cluster enumeration
[est_num_clusters, est_cluster_centroids, est_cluster_covmats] = performBayesianClusterEnumeration(data, num_features, num_samples_total, criterion, spec_num_clusters, num_clusters, plotting);
 