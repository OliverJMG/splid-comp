% ADD_NODE - Add a satellite Pattern-of-Life node to a list of nodes and 
% update a filtered structure to store information on the new node.
%
% INPUTS:
%   n: Node object to be added.
%   nodes: Cell array containing existing nodes.
%   longitudes: Array of longitudes.
%   inclinations: Array of inclinations.
%   filtered: Structure to store filtered information.
%
% OUTPUTS:
%   nodes: Updated cell array with the added node.
%   filtered: Updated structure with filtered information.
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

function [nodes,filtered] = add_node(n, nodes, longitudes, inclinations, filtered)

    nodes{end} = nodes{end}.char_mode(n.index, longitudes, inclinations);

    if strcmp(n.type, 'AD')
        nodes{end}.mtype = 'NK';
    end

    if ~strcmp(nodes{end}.mtype, 'NK')
        filtered.indices = [filtered.indices; nodes{end}.index, n.index];
        filtered.modes.SK = [filtered.modes.SK; nodes{end}.mtype];
        stop_NS = strcmp(n.type, 'ID');
        filtered.modes.end = [filtered.modes.end; stop_NS];
    end
    
    nodes{end+1} = n;
end