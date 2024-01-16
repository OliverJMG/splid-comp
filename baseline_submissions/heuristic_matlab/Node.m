% INDEX_DICT - MATLAB class definition for representing satellite 
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
 
classdef Node
    properties
        satcat
        t
        tstring
        t0
        t1
        neep
        dt
        index
        next_index
        type
        signal
        notes
        lon
        confidence
        correlated
        time_err
        mode_lons
        mode_incs
        EW_std
        NS_std
        mtype
    end
    
    methods
        function obj = Node(satcat, t, t0, t1, dt, index, next_index, ntype, signal, lon, confidence, mtype)
            obj.satcat = str2double(satcat);
            obj.t = t;
            obj.tstring = datestr(t, 'yyyy-mm-ddTHH:MM');
            
            if nargin < 3 || isempty(t0)
                t0 = t;
            end
            if nargin < 4 || isempty(t1)
                t1 = t0;
            end
            
            obj.t0 = t0;
            obj.t1 = t1;
            obj.neep = t0;
            obj.dt = dt;
            obj.index = index;
            obj.next_index = next_index;
            obj.type = ntype;
            obj.signal = signal;
            obj.notes = {};
            obj.lon = lon;
            obj.confidence = confidence;
            obj.correlated = false;
            obj.time_err = 0.0;
            obj.mode_lons = [];
            obj.mode_incs = [];
            obj.EW_std = [];
            obj.NS_std = [];
            obj.mtype = mtype;
        end
        
        function obj = char_mode(obj, next_index, lons, incs)
            obj.next_index = next_index;
            obj.mode_lons = lons(obj.index:obj.next_index);
            obj.mode_incs = incs(obj.index:obj.next_index);
            EW_db = max(obj.mode_lons) - min(obj.mode_lons);
            EW_sd = std(obj.mode_lons);
            EW = (EW_db - EW_sd) / EW_sd;
            obj.NS_std = std(incs);
            
            if strcmp(obj.type, 'ID')
                obj.mtype = 'NK';
            elseif strcmp(obj.type, 'AD')
                obj.mtype = 'NK';
            elseif strcmp(obj.type, 'IK')
                if EW < 5.1
                    obj.mtype = 'CK';
                else
                    obj.mtype = 'EK';
                end
            elseif strcmp(obj.type, 'ES')
                obj.mtype = 'ES';
            elseif strcmp(obj.type, 'SS')
                if EW < 5.1
                    obj.mtype = 'CK';
                else
                    obj.mtype = 'EK';
                end
            end
        end
        
        function description = describe(obj, ntype)
            obj.type = ntype;
            description = 'unknown';
            
            if strcmp(obj.type, 'ID')
                description = 'Initiate Drift';
            elseif strcmp(obj.type, 'IK')
                description = 'End Drift';
            elseif strcmp(obj.type, 'AD')
                description = 'Adjust Drift';
            end
        end
        
        function id = ID(obj)
            id = [obj.type, '.', num2str(obj.satcat), '@', datestr(obj.t, 'yyyy-mm-ddTHH:MM')];
            % Uncomment the following lines if dt is not always empty
            % if ~isempty(obj.dt)
            %     id = [obj.type, '.', num2str(obj.satcat), '@', datestr(obj.t0, 'yyyy-mm-ddTHH:MM'), '+', num2str(floor(obj.dt*24*60))];
            % else
            %     id = [obj.type, '.', num2str(obj.satcat), '@', datestr(obj.t0, 'yyyy-mm-ddTHH:MM')];
            % end
        end
        
        function note(obj, notes)
            obj.notes{end+1} = notes;
        end
        
        function see_notes(obj)
            disp(obj.ID());
            for i = 1:length(obj.notes)
                disp(['     ', obj.notes{i}]);
            end
        end
        
        function clear_notes(obj)
            obj.notes = {};
        end
        
        function correlate(obj, t)
            obj.correlated = true;
            obj.time_err = (t - obj.t) * 24 * 3600; % Convert to seconds
        end
    end
end