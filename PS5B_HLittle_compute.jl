#import packages we (may) use
using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV, Plots

#call the script
include("PS5B_HLittle_model.jl") 

#initialize
prim, mut = Initialize()
#run the function after initializing to fill in the q grid with relevant indicies by state
AfterInit(mut)

#fill in the best response function given the state from i's perspective
BR(prim, mut)

#iterate until you get the best response functions
BR_iterate(prim, mut; tol = 1e-4, err = 100.0)

#print the q_BR function so we know what's happening
println(mut.q_BR)
###I'm satisfied enough with how this looks to move forward...

#calculate the stage game profits
Profits_stage(prim, mut)

#print the π_BR matrix
println(mut.π_BR)

#try to solve for the optimal investment strategies
Iterate(prim, mut; tol1 = 0.0001, tol2 = 0.0001)

#=
Graveyard of failed and forgotten things


#iterate until you get the best response functions
BR_iterate(prim, mut; tol = 1e-4, err = 100.0)


=#