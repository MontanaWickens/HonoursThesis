#!/bin/bash

# sed 's/\r$//' auto.sh > auto_linux.sh
a=( 6 )                 #number of actions e.g., a = 11 refers to 10 harvest choices plus 'No Stock Assessment'
s=( 3 )                 #number of biomass intervals
K=( 300000 )            #maximum K value
r=( 2 )                 #number of intervals for proliferation rate
k=( 2 )                 #number of intervals for carrying capacity
costs=( 40000 )         #cost of completing a stock assessment
runs=( 1 )

for cost in "${costs[@]}"
do
    echo "generate_fishery_model_file.py" 
    #python generate_fishery_model_file.py -c "${cost}" -a "${a}" -s "${s}" -r "${r}" -k "${k}"
    echo "generate_world_model_file.py"
    #python generate_world_model_file.py -c "${cost}" -a "${a}" -s "${s}" -K "${K}" -r "${r}" -k "${k}"
    cd ..
    echo "Sending model to POMDP solver"
    ./pomdpx -m stockassessmentpomdp/a${a}-s${s}-r${r}-k${k}_modelA.pomdpx -w stockassessmentpomdp/a${a}-s${s}-r${r}-k${k}_modelB.pomdpx -t 0.1 --runs ${runs} > "doublemodel_output.txt"
    echo "POMDP output saved to doublemodel_output.txt"
    cd stockassessmentpomdp
    echo "Reading POMDP output, saving plots"
    python read_double_model_output.py -c "${cost}"
done