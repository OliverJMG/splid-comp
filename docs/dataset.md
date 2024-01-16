# Satellite Pattern-of-Life Identification Dataset (SPLID)

## Download the Dataset
The challenge dataset can be downloaded from <a href="https://www.dropbox.com/scl/fo/jt5h1f82iycjb8elybmlz/h?rlkey=bjcmny486ddf7m0j7b9uok9ww&dl=0">here</a>.

## Dataset Description
SPLID consists of a public challenge dataset and a private evaluation dataset. The public challenge dataset contains 500 simulated satellite trajectories, each operating under different mission objectives and equipped with different propulsion capabilities. The trajectories are generated using a high-fidelity astrodynamics simulator and are designed to represent a diverse set of satellite behaviors. The private evaluation dataset contains additional data that will be used to evaluate the performance of the participants' models. This private evaluation dataset consists of both simulated satellite trajectories and historical satellite trajectories generated from VCM data and high-accuracy ephemerides provided by satellite operators.

The dataset consists of astrometric data over six months at a two-hour temporal resolution. The astrometric data consists of the osculating orbital elements, geographic positions, and the Cartesian states of the satellite in the J2000 inertial reference frame. The figure below shows a snapshot of the data for a single satellite.

![alt text](example_data.png)

Each data is accompanied by a list of expert-annotated time-stamped pattern-of-life (PoL) nodes. Each row in the satellite PoL node list begins with a time index of the node, followed by the direction ("EW" for east-west and "NS" for north-south), the node type, and the propulsion type used while in this behavioral mode. The time index refers to the row index of the astrometric data that corresponds to that particular node. The description of each label is provided in the table below.

| Node Label  | Description | 
| ------------- | ------------- | 
| SS  | Start of the study period  | 
| ES  | End of the study period  | 
| ID  | Initiate drift  | 
| AD  | Adjust drift  |
| IK  | Initiate station-keeping  | 

| Type Label  | Description |
| ------------- | ------------- |
| NK | Not station-keeping |
| CK | Station-keeping using chemical propulsion system |
| EK | Station-keeping using electric propulsion system |
| HK | Station-keeping using hybrid propulsion system |
