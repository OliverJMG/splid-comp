% heuristic_baseline - This Matlab code will guide the reader on
% implementing the Satellite Node Identification and Classification Tool
% (SNICT) as a heuristic method to identify and classify the satellite
% Pattern-of-Life nodes.
%
% Based on the Python code by Liz Solera, 2023
% Solera, H. E., T. G. Roberts, and R. Linares. "Geosynchronous Satellite 
% Pattern of Life Node Detection and Classification." 9th Space Traffic 
% Management Conference, Austin, TX. 2023.
%
% Copyright (C) 2023 by Peng Mun Siew
%
% This code is licensed under the MIT License.
%
% Author: Peng Mun Siew
% Massachusetts Institute of Technology, Dept. of Aeronautics and Astronautics
% email: siewpengmun@yahoo.com
% Dec 2023; Last revision: 5-Dec-2023

clearvars
clc
close all

%%
% We will be using custom classes to store the node information
detected = index_dict();
filtered = index_dict();

% Initialize datalist as an empty cell array
datalist = {};

% Searching for training data within the dataset folder
files = dir('../../dataset/train/*.csv');
for i = 1:length(files)
    datalist = [datalist; fullfile('../../dataset/train/', files(i).name)];
end

% Sort the training data and labels
labeldir = '../../dataset/labels.csv';

% Print the sorted filepath to the training data
disp(datalist);

%%
% Load the fifth training data. 
idx_data = 5;

data_path = datalist{idx_data};
data = readtable(data_path);

%%
% The SNICT uses longitudinal and inclination information to detect and
% characterize the changes in a satellite's behavioral mode. The
% longitudinal and inclination information are first extracted from the 
% data.

% Read the ObjectID from the filename
satcat = data_path(end-6:end-4);

% Extracting longitudinal and inclination information from the table
longitudes = data.Longitude_deg_;
inclinations = data.Inclination_deg_;

% Arbitrary start and end times
starttime = datetime("2023-01-01 00:00:00", "Format", "yyyy-MM-dd HH:mm:ss", "TimeZone", "UTC");
endtime = datetime("2023-07-01 00:00:00", "Format", "yyyy-MM-dd HH:mm:ss", "TimeZone", "UTC");

%%
% We will first identify possible East-West Pattern-of-Life nodes by
% analyzing the changes in longitudinal values.

% Get std for longitude over a 24 hours window
lon_std = zeros(size(data.Longitude_deg_));
nodes = [];
steps_per_day = 12;
lon_was_baseline = true;
lon_baseline = 0.03;

for i = 1:length(data.Longitude_deg_)
    if i <= steps_per_day+1
        lon_std(i) = std(data.Longitude_deg_(1:steps_per_day),1);
    else
        lon_std(i) = std(data.Longitude_deg_(i-steps_per_day:i-1),1);
    end
end
% Node(satcat, t, t0, t1, dt, index, next_index, ntype, signal, lon, confidence, mtype)
ssEW = Node(satcat, starttime, [], [], [], 1, [], 'SS', 'EW', [], [], []);
es = Node(satcat, endtime, [], [], [], length(data.Longitude_deg_), [], 'ES', 'ES', [], [], 'ES');

% Run LS detection
for i = steps_per_day+1:length(lon_std)-steps_per_day
    % if at least 1 day has elapsed since t0
    max_lon_std_24h = max(lon_std(i-steps_per_day:i-1));
    min_lon_std_24h = min(lon_std(i-steps_per_day:i-1));
    A = abs(max_lon_std_24h - min_lon_std_24h) / 2;
    th_ = 0.95 * A;
    
    % ID detection
    if (lon_std(i) > lon_baseline) && lon_was_baseline
        % if sd is elevated & last sd was at baseline
        before = mean(data.Longitude_deg_(i-steps_per_day:i-1));  % mean of previous day's longitudes
        after = mean(data.Longitude_deg_(i:i+steps_per_day-1));   % mean of next day's longitudes
        % if not temporary noise, then real ID
        if abs(before - after) > 0.3  % if means are different
            lon_was_baseline = false;  % the sd is not yet back at baseline
            index = i;
            if i < steps_per_day+2
                ssEW.mtype = 'NK';
            else
                detected.times.ID = [detected.times.ID; starttime + hours((i-1)*2)];
            end
        end
    % IK detection
    elseif (lon_std(i) <= lon_baseline) && (~lon_was_baseline)
        % elif sd is not elevated and drift has already been initialized
        drift_ended = true;  % toggle end-of-drift boolean 
        for j = 1:steps_per_day
            % for the next day, check if the longitude changes from the current value
            if abs(data.Longitude_deg_(i) - data.Longitude_deg_(i+j)) > 0.3
                drift_ended = false;  % the drift has not ended
            end
        end
        if drift_ended  % if the drift has ended
            lon_was_baseline = true;  % the sd is back to baseline
            detected.times.IK = [detected.times.IK; starttime + hours((i-1)*2)];  % save tnow as end-of-drift
            detected.indices = [detected.indices; index, i];  % save indices for t-start & t-end
        end
    % Last step
    elseif (i == (length(lon_std)-steps_per_day-1)) && (~lon_was_baseline)
        detected.times.IK = [detected.times.IK; starttime + hours((i-1)*2)];
        detected.indices = [detected.indices; index, i];
    % AD detection
    elseif ((lon_std(i) - max_lon_std_24h > th_) || (min_lon_std_24h - lon_std(i) > th_)) && (~lon_was_baseline)
        % elif sd is elevated and drift has already been initialized
        if i > steps_per_day+3
            detected.times.AD = [detected.times.AD; starttime + hours((i-1)*2)];
            detected.AD_dex = [detected.AD_dex; i];
        end
    end
end

%%
% Next, we will filter the East-West nodes and merge nearby nodes with the
% same label base on their heuristics.

toggle = true;
nodes{end+1} = ssEW;

if length(detected.times.IK) == 1
    if length(detected.times.ID) == 1
        % keep the current ID
        filtered.times.ID = [filtered.times.ID; detected.times.ID(1)];

        ID = Node(satcat, detected.times.ID(1), [], [], [], detected.indices(1, 1), [], 'ID', 'EW', longitudes(detected.indices(1, 1)), [], [] );
        [nodes,filtered ] = add_node(ID, nodes, longitudes, inclinations, filtered);
    end
    
    filtered.times.IK = [filtered.times.IK; detected.times.IK(1)];

    IK = Node(satcat, detected.times.IK(1), [], [], [], detected.indices(1, 2), [], 'IK', 'EW', longitudes(detected.indices(1, 2)), [], []);
    apnd = true;
    
    if length(detected.times.AD) == 1
        AD = Node(satcat, detected.times.AD(1), [], [], [], detected.AD_dex(1), [], 'AD', 'EW', [], [], []);
        [nodes,filtered ] = add_node(AD, nodes, longitudes, inclinations, filtered);
    elseif length(detected.times.AD) > 1
        for j=1:length(detected.times.AD)
            ad = Node(satcat, detected.times.AD(j), [], [], [], detected.AD_dex(j), [], 'AD', 'EW', [], [], []);
            if j < length(detected.times.AD)
                ad_next = Node(satcat, detected.times.AD(j+1), [], [], [], detected.AD_dex(j+1), [], 'AD', 'EW', [], [], []);
            else 
                ad_next = [];
            end
            
            if ad.t > starttime + hours((detected.indices(1, 1)-1)*2) && ad.t < IK.t
                if apnd && ~isempty(ad_next)
                    if (ad_next.t - ad.t) > hours(24)
                        [nodes,filtered] = add_node(ad, nodes, longitudes, inclinations, filtered);
                    else
                        [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                        apnd = false;
                    end
                elseif apnd && isempty(ad_next)
                    [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                elseif ~apnd && ~isempty(ad_next)
                    if (ad_next.t - ad.t) > hours(24)
                        apnd = true;
                    end
                end
            end
            j = j + 1;
        end
    end
    
    if detected.indices(1, 2) ~= length(lon_std) - steps_per_day - 1
        [nodes,filtered ] = add_node(IK, nodes, longitudes, inclinations, filtered);
    end
end

for i = 1:length(detected.times.IK)-1
    if toggle
        if (starttime + hours((detected.indices(i+1, 1)-1)*2) - detected.times.IK(i)) > hours(36)
            filtered.times.ID = [filtered.times.ID; detected.times.ID(i)];
            filtered.times.IK = [filtered.times.IK; detected.times.IK(i)];
            ID = Node(satcat, detected.times.ID(i), [], [], [], detected.indices(i, 1), [], 'ID', 'EW', longitudes(detected.indices(i, 1)), [], []);
            [nodes,filtered ] = add_node(ID, nodes, longitudes, inclinations, filtered);
            IK = Node(satcat, detected.times.IK(i), [], [], [], detected.indices(i, 2), [], 'IK', 'EW', longitudes(detected.indices(i, 2)), [], [] );
            apnd = true;
            for j = 1:length(detected.times.AD)
                
                ad = Node(satcat, detected.times.AD(j), [], [], [], detected.AD_dex(j), [], 'AD', 'EW', [], [], []);
                if j < length(detected.times.AD)-1
                    ad_next = Node(satcat, detected.times.AD(j+1), [], [], [], detected.AD_dex(j+1), [], 'AD', 'EW', [], [], []);
                else 
                    ad_next = [];
                end
                if (ad.t > ID.t) && (ad.t < IK.t)
                    if apnd && ~isempty(ad_next)
                        if (ad_next.t - ad.t) > hours(24)
                            [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                        else
                            [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                            apnd = false;
                        end
                    elseif apnd && isempty(ad_next)
                        [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                    elseif ~apnd && ~isempty(ad_next)
                        if (ad_next.t - ad.t) > hours(24)
                            apnd = true;
                        end
                    end
                end
            end
            if detected.indices(1, 2) ~= (length(lon_std) - steps_per_day - 1)
                [nodes,filtered ] = add_node(IK, nodes, longitudes, inclinations, filtered);
            end
            if i == length(detected.times.IK)-1
                filtered.times.ID = [filtered.times.ID; starttime + hours((detected.indices(i+1, 1)-1)*2)];
                

                ID = Node(satcat, starttime + hours((detected.indices(i+1, 1)-1)*2), [], [], [], detected.indices(i+1, 1), [], 'ID', 'EW', longitudes(detected.indices(i+1, 1)), [], []);
                [nodes,filtered ] = add_node(ID, nodes, longitudes, inclinations, filtered);
                IK = Node(satcat, detected.times.IK(i+1), [], [], [], detected.indices(i+1, 2), [], 'IK', 'EW', longitudes(detected.indices(i+1, 2)), [], []);
                apnd = true;
                for j = 1:length(detected.times.AD)
                    ad = Node(satcat, detected.times.AD(j), [], [], [], detected.AD_dex(j), [], 'AD', 'EW', [], [], []);
                    if j < length(detected.times.AD)-1
                        ad_next = Node(satcat, detected.times.AD(j+1), [], [], [], detected.AD_dex(j+1), [], 'AD', 'EW', [], [], []);
                    else 
                        ad_next = [];
                    end
                    if (ad.t > ID.t) && (ad.t < IK.t)
                        if apnd && ~isempty(ad_next)
                            if (ad_next.t - ad.t) > hours(24)
                                [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                            else
                                [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                                apnd = false;
                            end
                        elseif apnd && isempty(ad_next)
                            [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                        elseif ~apnd && ~isempty(ad_next)
                            if (ad_next.t - ad.t) > hours(24)
                                apnd = true;
                            end
                        end
                    end
                end
                if detected.indices(i, 2) ~= (length(lon_std) - steps_per_day - 1)
                    filtered.times.IK = [filtered.times.IK; detected.times.IK(i+1)];
                    [nodes,filtered ] = add_node(IK, nodes, longitudes, inclinations, filtered);
                end
            end
        else
            ID = Node(satcat, detected.times.ID(i), [], [], [], detected.indices(i, 1), [], 'ID', 'EW', longitudes(detected.indices(i, 1)), [], []);
            [nodes,filtered ] = add_node(ID, nodes, longitudes, inclinations, filtered);
            AD = Node(satcat, detected.times.IK(i), [], [], [], detected.indices(i, 2), [], 'AD', 'EW', longitudes(detected.indices(i, 2)), [], []);
            IK = Node(satcat, detected.times.IK(i+1), [], [], [], detected.indices(i+1, 2), [], 'IK', 'EW', longitudes(detected.indices(i+1, 2)), [], []);
            [nodes,filtered ] = add_node(AD, nodes, longitudes, inclinations, filtered);
            apnd = true;
            for j = 1:length(detected.times.AD)
                ad = Node(satcat, detected.times.AD(j), [], [], [], detected.AD_dex(j), [], 'AD', 'EW', [], [], []);
                if j < length(detected.times.AD)-1
                    ad_next = Node(satcat, detected.times.AD(j+1), [], [], [], detected.AD_dex(j+1), [], 'AD', 'EW', [], [], []);
                else 
                    ad_next = [];
                end
                if (ad.t > ID.t) && (ad.t < IK.t)
                    if apnd && ~isempty(ad_next)
                        if (ad_next.t - ad.t) > hours(24)
                            [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                        else
                            [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                            apnd = false;
                        end
                    elseif apnd && isempty(ad_next)
                        [nodes,filtered ] = add_node(ad, nodes, longitudes, inclinations, filtered);
                    elseif ~apnd && ~isempty(ad_next)
                        if (ad_next.t - ad.t) > hours(24)
                            apnd = true;
                        end
                    end
                end
            end
            if detected.indices(1, 2) ~= (length(lon_std) - steps_per_day - 1)
                [nodes,filtered ] = add_node(IK, nodes, longitudes, inclinations, filtered);
            end
            filtered.times.ID = [filtered.times.ID; detected.times.ID(i)];
            filtered.times.AD = [filtered.times.AD; detected.times.IK(i)];
            filtered.times.IK = [filtered.times.IK; detected.times.IK(i+1)];
            toggle = false;
        end
    else
        toggle = true;
    end
end
[nodes,filtered ] = add_node(es, nodes, longitudes, inclinations, filtered);
%%
% Identify possible North-South Pattern-of-Life nodes by analyzing the
% changes in the inclination values.

ssNS = Node(satcat, starttime, [], [], [], 0, [], 'SS', 'NS', [], [], []);
for j = 1:size(filtered.indices,1)
    indices = filtered.indices(j, :);
    first = indices(1) == 1;
    times = [];
    dexs = [];
    inc = inclinations(indices(1):indices(2));
    t = (indices(1):indices(2)) * 2;
    rate = (steps_per_day / (indices(2) - indices(1))) * (max(inc) - min(inc));
    XIPS_inc_per_day = 0.0005; % 0.035/30
    if (rate < XIPS_inc_per_day) && (indices(1) < steps_per_day) && (indices(2) > steps_per_day)
        if filtered.modes.end(j)
            nodes{end+1} =  Node(satcat, starttime + hours((indices(2)-1) * 2), [], [], [], indices(2), [], 'ID', 'NS', [], [], 'NK');
        end
        ssNS.mtype = filtered.modes.SK(j,:);
    elseif (rate < XIPS_inc_per_day)
        nodes{end+1} = Node(satcat, times(1), [], [], [], dexs(1), [], 'IK', 'NS', [], [], filtered.modes.SK(j,:));
        if filtered.modes.end(j)
            nodes{end+1} = Node(satcat, starttime + hours((indices(2)-1) * 2), [], [], [], indices(2), [], 'ID', 'NS', [], [], 'NK');
        end
    else
        dt = zeros(1, length(inc) - 1);
        for i = 1:length(inc) - 1
            dt(i) = (inc(i + 1) - inc(i)) / (2 * 60 * 60);
        end
        prev = 1.0;
        for i = 1:length(dt) - 1
            if abs(dt(i)) > 5.5e-7
                times = [times, starttime + hours(t(i))];
                dexs = [dexs, i + indices(1)];
                if (abs(mean(inc(1:i)) - mean(inc(i:end))) / std(inc(1:i))) / prev < 1.0
                    if first && length(times) == 2
                        ssNS.mtype = filtered.modes.SK(1,:);
                        first = false;
                    end
                elseif length(times) == 2
                    first = false;
                end
                prev = abs(mean(inc(1:i)) - mean(inc(i:end))) / std(inc(1:i));
            end
        end
        if ~isempty(times)
            nodes{end+1} = Node(satcat, times(1), [], [], [], dexs(1), [], 'IK', 'NS', [], [], filtered.modes.SK(j,:));
            if filtered.modes.end(j)
                nodes{end+1} = Node(satcat, starttime + hours((indices(2)-1) * 2), [], [], [], indices(2), [], 'ID', 'NS', [], [], 'NK');
            end
            ssNS.mtype = 'NK';
        elseif filtered.indices(1, 1) == 1
            ssNS.mtype = filtered.modes.SK(1,:);
        end
    end
end
nodes{end+1} = ssNS;

%%
% Lastly, we will tidy up the output of the SNICT algorithm to match the
% SPLID data format.

ObjectID = [];
TimeIndex = [];
Direction = [];
Node = [];
Type = [];

for i = 1:length(nodes)
    ObjectID = [ObjectID; nodes{i}.satcat];
    TimeIndex = [TimeIndex; hours(nodes{i}.t - starttime) / 2 ];
    Direction = [Direction; nodes{i}.signal];
    Node = [Node; nodes{i}.type];
    Type = [Type; nodes{i}.mtype];
end

% Create a table
prediction = table(ObjectID, TimeIndex, Direction, Node, Type);
prediction = sortrows(prediction,'TimeIndex');
% Display the table
disp(prediction);

%%
% Save the prediction table to a CSV file

writetable(prediction, 'prediction.csv');