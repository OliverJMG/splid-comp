% INDEX_DICT - MATLAB class definition for storing satellite Pattern-of-Life index information.
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

classdef index_dict
    properties
        times
        indices
        AD_dex
        modes
    end
    
    methods
        function obj = index_dict()
            obj.times = obj.IDADIK();
            obj.indices = [];
            obj.AD_dex = [];
            obj.modes = obj.mode();
        end
    end
    
    methods (Static)
        function result = IDADIK()
            result.ID = [];
            result.AD = [];
            result.IK = [];
        end

        function result = mode()
            result.SK = [];
            result.end = [];
        end
    end
end
