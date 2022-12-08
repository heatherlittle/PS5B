#import packages we (may) use
using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, CSV, Plots

#create a primitive struct
@with_kw struct Primitives

    δ::Float64 = 0.1
    β::Float64 = 1/1.05
    α::Float64 = 0.06
    a::Float64 = 40 #note that there is a typo in the Pset and that this differs a bit from the slides
    b::Float64 = 10 #we want to eventually get P = 4 - (1/10)*Q
    A::Float64 = 4
    B::Float64 = 1/10

    #S::Int64 = 100 #the number of states 10x10, don't use this, rather 
    CapGrid::Array{Int64, 1} = [0; 5; 10; 15; 20; 25; 30; 35; 40; 45]

end #close the primitives struct

mutable struct Mutable

    #will likely not use these but rather use the matrix below
    q::Array{Int64, 2} #each row of this will determine the state ω = q1, q2
    their_state::Array{Int64, 1} #if an index indicates which state I'M in, I want to be able to map my state to the perspective of my opponent (the index of the state they're in)
    qstar::Array{Int64, 1} #this is the best response quanity for a given state
    prof_func::Array{Float64, 1} #this is the profit associated with each state of the stage game, analogus to a value function we'll solve with iteration

    #let the element i,j be the state in which player 1 has the ith element of the CapGrid and player 2 has the jth element of the CapGrid
    ###we will fill in the optimal quantity/q*/best response and the associated profit for each
    q_BR::Array{Float64, 2}
    π_BR::Array{Float64, 2}

    pol_func::Array{Float64, 2}
    val_func::Array{Float64, 2}

end #close mutable struct

function Initialize()
    prim = Primitives()

    #a lot of this won't be used, 10 by 10 matrix is easier
    q = zeros(100, 2) #I will later fill this in with the relevant indicies of q1, q2 for all possible states
    their_state = zeros(100)
    qstar = zeros(100) #this will be filled in with the optimal quantity chosen in each state
    prof_func = zeros(100)

    q_BR = zeros(10,10)
    π_BR = zeros(10,10)

    pol_func = zeros(10,10)
    val_func = zeros(10,10)

    mut = Mutable(q, their_state, qstar, prof_func, q_BR, π_BR, pol_func, val_func)
    return prim, mut

end #close the initialize function

#I think it may be easier to do this as a matrix, so I'm ditching it
function AfterInit(mut::Mutable)
    @unpack q, their_state = mut

    #Fill in the first column as "my capacity"
    q[1:10, 1] .= 1
    q[11:20, 1] .= 2
    q[21:30, 1] .= 3
    q[31:40, 1] .= 4
    q[41:50, 1] .= 5
    q[51:60, 1] .= 6
    q[61:70, 1] .= 7
    q[71:80, 1] .= 8
    q[81:90, 1] .= 9
    q[91:100, 1] .= 10

    #Fill in the second column as "your capacity"
    for i = 1:10
        q[i, 2] = i 
    end
    for i = 1:10
        q[10+i, 2] = i 
    end
    for i = 1:10
        q[20+i, 2] = i 
    end
    for i = 1:10
        q[30+i, 2] = i 
    end
    for i = 1:10
        q[40+i, 2] = i 
    end
    for i = 1:10
        q[50+i, 2] = i 
    end
    for i = 1:10
        q[60+i, 2] = i 
    end
    for i = 1:10
        q[70+i, 2] = i 
    end
    for i = 1:10
        q[80+i, 2] = i 
    end
    for i = 1:10
        q[90+i, 2] = i 
    end

    #fill in the their_state vector to get a mapping from the state you're in to the state of your opponent
    for i = 1:10
        their_state[i] = 1+(i-1)*10
    end
    for i = 1:10
        their_state[10+i] = 2+(i-1)*10
    end
    for i = 1:10
        their_state[20+i] = 3+(i-1)*10
    end
    for i = 1:10
        their_state[30+i] = 4+(i-1)*10
    end
    for i = 1:10
        their_state[40+i] = 5+(i-1)*10
    end
    for i = 1:10
        their_state[50+i] = 6+(i-1)*10
    end
    for i = 1:10
        their_state[60+i] = 7+(i-1)*10
    end
    for i = 1:10
        their_state[70+i] = 8+(i-1)*10
    end
    for i = 1:10
        their_state[80+i] = 9+(i-1)*10
    end
    for i = 1:10
        their_state[90+i] = i*10
    end

end #close the function


function BR(prim::Primitives, mut::Mutable) #note that I'm filling in the best response from i's perspective
    @unpack A, B = prim
    #note this refers to the three cases of NE in slide 20, typo on case three should be greater than or equal to

    holder_q_BR = zeros(10,10) #create a holder

    #given A=4 and B=1/10, when both player's have qbar less than or eq to 13.333, produce A/(3B)
    for i= 1:10
        for j = 1:10
            #use the indicies and the capital grid to pull out each player's actual capacity
            qbar_i = prim.CapGrid[i]
            qbar_j =prim.CapGrid[j]

            holder_q_BR[i,j] = min(qbar_i, ((A/(2*B))-(mut.q_BR[j,i]/2)))

        end #close loop over j
    end #close loop over i

    return holder_q_BR 

end #close function

function BR_iterate(prim::Primitives, mut::Mutable; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        q_BR_next = BR(prim, mut) #spit out new BR matrix
        err = abs.(maximum(q_BR_next.-mut.q_BR))/abs(q_BR_next[10,10]) #reset error level
        ####this equation above is the sup norm written out, it's how we're calculating the error
        mut.q_BR = q_BR_next #update value function, saving val func as the guess you made with v_next
        n+=1
    end
    println("Best response function converged in ", n, " iterations.")
end #this is just like VFI, since my best response is a function of the best response function...


function Profits_stage(prim::Primitives, mut::Mutable)
    @unpack A, B = prim

    #simply use the best response, q*, to determine profit in each state
    #will record this from the perspective of i, but symmetric profits for j
    for i = 1:10
        for j = 1:10
            my_q = mut.q_BR[i,j]
            your_q = mut.q_BR[j,i] #since strategies are symmetric

            mut.π_BR[i,j] = (A - B*my_q - B*your_q)*my_q

        end #close loop over j
    end #close loop over i

    #not going to return anything because I've updated the mutable struct

end #close function to fill in the stage game profits

function trans(x::Float64) #this is a function of investment, note we will also call the primitives but those aren't technically an input

    vec = zeros(3) #initialize a vector in which we will store the probabilities of moving up
    #probability that my capacity goes up 1
    vec[1] = ((1-prim.δ)*prim.α*x)/(1+prim.α*x)
    #probability that my capacity stays the same 
    vec[2] = ((1-prim.δ)+prim.δ*prim.α*x)/(1+prim.α*x)
    #probability that my capacity goes down 1
    vec[3] = (prim.δ)/(1+prim.α*x)

    return vec
    #return vec[1], vec[2], vec[3] #this will allow me to call a specific transition probability

end #close the function for transitions

function Next_Val(prim::Primitives, mut::Mutable)

    V_next = zeros(10,10) #initialize a value function

    for i = 1:10
        for j = 1:10

            #using the current policy function, pull out the investment choices of you and your opponent
            x_me = mut.pol_func[i,j]
            x_you = mut.pol_func[j,i]

            #using the investment associated with the current policy function, create a transition matrix, 3x3 
            ###this mat will move you around the current i,j in a the -1, 0 +1 movement combos of you and your opponent
            ###to make sense of this, draw out a matrix and think about the probability(+,0,- | my investment) x probability(+,0,- | your investment) 
            trans_now = zeros(3,3) #initialize the i,j specific transition matrix
            ###note that index 1 means moving +, index 2 means moving 0, index 3 means moving -
            trans_now[1,1] = trans(x_me)[3]*trans(x_you)[1]
            trans_now[1,2] = trans(x_me)[2]*trans(x_you)[1]
            trans_now[1,3] = trans(x_me)[1]*trans(x_you)[1]
            trans_now[2,1] = trans(x_me)[3]*trans(x_you)[2]
            trans_now[2,2] = trans(x_me)[2]*trans(x_you)[2]
            trans_now[2,3] = trans(x_me)[1]*trans(x_you)[2]
            trans_now[3,1] = trans(x_me)[3]*trans(x_you)[3]
            trans_now[3,2] = trans(x_me)[2]*trans(x_you)[3]
            trans_now[3,3] = trans(x_me)[1]*trans(x_you)[3]

            #this is a little bit gross, but I'm elemnt wise multiplying the existing value function (3X3 centered at i,j) and the relevant transition matrix, using the current investment decisions of me and my opponent
            ###actually this is stupid gross because I need to account for edge cases
            if i == 1
                if j == 1
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i, j]*trans_now[1,1] + mut.val_func[i, j]*trans_now[1,2] + mut.val_func[i, j+1]*trans_now[1,3] + mut.val_func[i,j]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j+1]*trans_now[2,3] + mut.val_func[i+1, j]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j+1]*trans_now[3,3])
                elseif j == 10
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i, j-1]*trans_now[1,1] + mut.val_func[i, j]*trans_now[1,2] + mut.val_func[i, j]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j]*trans_now[2,3] + mut.val_func[i+1, j-1]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j]*trans_now[3,3])
                else 
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i, j-1]*trans_now[1,1] + mut.val_func[i, j]*trans_now[1,2] + mut.val_func[i, j+1]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j+1]*trans_now[2,3] + mut.val_func[i+1, j-1]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j+1]*trans_now[3,3])
                end #close the inner if statement
            elseif i == 10
                if j == 1
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j+1]*trans_now[1,3] + mut.val_func[i,j]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j]*trans_now[2,3] + mut.val_func[i, j]*trans_now[3,1] + mut.val_func[i, j]*trans_now[3,2] + mut.val_func[i, j+1]*trans_now[3,3])
                elseif j == 10
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j-1]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j]*trans_now[2,3] + mut.val_func[i, j-1]*trans_now[3,1] + mut.val_func[i, j]*trans_now[3,2] + mut.val_func[i, j]*trans_now[3,3])
                else 
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j-1]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j+1]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j+1]*trans_now[2,3] + mut.val_func[i, j-1]*trans_now[3,1] + mut.val_func[i, j]*trans_now[3,2] + mut.val_func[i, j+1]*trans_now[3,3])
                end #close the inner if statement
            else
                if j ==1
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j+1]*trans_now[1,3] + mut.val_func[i,j]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j+1]*trans_now[2,3] + mut.val_func[i+1, j]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j+1]*trans_now[3,3])
                elseif j == 10
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j-1]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j]*trans_now[2,3] + mut.val_func[i+1, j-1]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j]*trans_now[3,3])
                else 
                    V_next[i,j] = mut.π_BR[i,j] - x_me + prim.β*(mut.val_func[i-1, j-1]*trans_now[1,1] + mut.val_func[i-1, j]*trans_now[1,2] + mut.val_func[i-1, j+1]*trans_now[1,3] + mut.val_func[i,j-1]*trans_now[2,1] + mut.val_func[i,j]*trans_now[2,2] + mut.val_func[i, j+1]*trans_now[2,3] + mut.val_func[i+1, j-1]*trans_now[3,1] + mut.val_func[i+1, j]*trans_now[3,2] + mut.val_func[i+1, j+1]*trans_now[3,3])
                end #close the inner if statement
             end #close the if else loop
        end #close the loop over j
    end #close the loop over i

    return V_next

end #close the bellman-ish / next value function

function Pol_Update(prim::Primitives, mut::Mutable)
    @unpack β, α, δ = prim

    W = zeros(10,10) #initialize W
    #first fill in W
    for i = 1:10
        for j = 1:10

            ###first, calculate the continuation value
            #since we're taking perspective of i, we have to be thinking about what j is doing
            x_you = mut.pol_func[j,i] #we will use this x to calculate the relevant transitions
            if j == 1
                W[i,j] = trans(x_you)[1]*mut.val_func[i, j] + trans(x_you)[2]*mut.val_func[i, j] + trans(x_you)[3]*mut.val_func[i, j+1]
            elseif j == 10
                W[i,j] = trans(x_you)[1]*mut.val_func[i, j-1] + trans(x_you)[2]*mut.val_func[i, j] + trans(x_you)[3]*mut.val_func[i, j]
            else
                W[i,j] = trans(x_you)[1]*mut.val_func[i, j-1] + trans(x_you)[2]*mut.val_func[i, j] + trans(x_you)[3]*mut.val_func[i, j+1]
            end #close if else statement
        end #close loop over j
    end #close loop over i

    X = zeros(10,10)
    #then use W to update the pol_func, X
    for i = 1:10
        for j = 1:10

            #literally just the the formula given in the PSet, but with if statements to keep you in bounds
            if i == 1
                X[i,j] = max(0, ((1/α)*(-1+sqrt(β*α*((1-δ)*(W[i+1,j]-W[i,j])+δ*(W[i,j]-W[i,j]))))))
            elseif i == 10
                X[i,j] = max(0, ((1/α)*(-1+sqrt(β*α*((1-δ)*(W[i,j]-W[i,j])+δ*(W[i,j]-W[i-1,j]))))))
            else
                X[i,j] = max(0, ((1/α)*(-1+sqrt(β*α*((1-δ)*(W[i+1,j]-W[i,j])+δ*(W[i,j]-W[i-1,j]))))))
            end #close the if else loop 

        end #close loop over j
    end #close loop over i

    return X #the updated policy function

end #close the function to update the policy function

function Iterate(prim::Primitives, mut::Mutable; tol1::Float64 = 0.0001, tol2::Float64 = 0.0001)

    n = 0 #start with a counter
    #set our original policy and value function using the initialized mutable, they will be updated in the while loop below
    x_0 = mut.pol_func
    v_0 = mut.val_func

    #initialize errors
    error_x = 100
    error_v = 100
    #using the sup norm for our earlier VFI functions
    while error_x > tol1 && error_v > tol2

        #update the value and policy function
        x_1 = Pol_Update(prim, mut)
        v_1 = Next_Val(prim, mut)

        error_x = abs.(maximum(x_1.-x_0))/abs(x_1[10,10])
        error_v = abs.(maximum(v_1.-v_0))/abs(v_1[10,10]) 

        #store the new in mutable struct and make new old
        mut.pol_func = x_1
        mut.val_func = v_1
        x_0 = x_1
        v_0 = v_1

        n += 1 #update counter

    end #close the while loop

    println("We converged after ", n, " iterations.")

end #close the iteration process






#=
Graveyard of things I've decided are failures

    states::Array{Float64, 2} = [0 0; 0 5; 0 10; 0 15; 0 20; 0 25; 0 30; 0 35; 0 40; 0 45; 5 5; 5 10; 5 15; 5 20; 5 25; 5 30; 5 35; 5 40; 5 45; 10 10; 10 15; 10 20; ]


#the nash equilibrium in question 2 is just a basic cournot that's state dependent
###note that this is the stage game in step 2 on slide 22 of the notes (version published for our year)

function Stage_likeBellman(prim::Primitives, mut::Mutable)
    @unpack A, B = prim

    #initialize a vector we will spit out, like "value" in Bellman
    choice = zeros(100)

    #loop over every state
    for m = 1:100
        s = mut.their_state[m] #this gives the index of the state your opponent is in, we'll use it to get their best response
        #use indexing of state to pull relevant quanities out of the capital grid
        my_q = prim.CapGrid[mut.q[m, 1]]
        #your_q = prim.CapGrid[mut.q[m, 2]]
        ###I care about the chosen (best response) q of my opponent, not their capcity per say

        #you have two choices, either to produce at capacity or to produce at the interior solution
        ###here A=4 and B =1/10, we refer to the interior or capacity solutions on slide 20
        choice[m] = min(my_q, (80-(mut.qstar[s]/2))) #using my OPPONENT's best response to pick my best response

        #fill in the profit given the state you're in and everyone's best responses
        mut.prof_func[m] = (A-B*choice[m]-B*mut.qstar[s])*choice[m]
        #if that's not profitable, pull out of the market:
        if mut.prof_func[m] < 0
            choice[m] = 0
        end #close if statement

    end #close loop over states

    return choice

end #close function, like a value function almost

#Value function iteration
function BR_iterate(prim::Primitives, mut::Mutable; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        q_next = Stage_likeBellman(prim, mut) #spit out new vectors
        err = abs.(maximum(q_next.-mut.qstar))/abs(q_next[100]) #reset error level
        ####this equation above is the sup norm written out, it's how we're calculating the error
        mut.qstar = q_next #update value function, saving val func as the guess you made with v_next
        n+=1
    end
    println("Best response function converged in ", n, " iterations.")
end


function BR(prim::Primitives, mut::Mutable) #note that I'm filling in the best response from i's perspective
    @unpack A, B = prim
    #note this refers to the three cases of NE in slide 20, typo on case three should be greater than or equal to

    #given A=4 and B=1/10, when both player's have qbar less than or eq to 13.333, produce A/(3B)
    for i= 1:10
        for j = 1:10
            #use the indicies and the capital grid to pull out each player's actual capacity
            qbar_i = prim.CapGrid[i]
            qbar_j =prim.CapGrid[j]

            mut.q_BR[i,j] = min(qbar_i, ((A/(2*B))-(q_BR[j,i]/2)))

            #=
            ####use the if statement to loop over the three cases
            #uncosntrained case ##I think there's a typo and that this should be > not < ???
            if qbar_i > A/(3*B) && qbar_j > A/(3*B)
                mut.q_BR[i,j] = A/(3*B) #note, this doesn't make much sense-- how can we produce more than we have the capacity to produce
        
            #constrained case
            elseif min(qbar_i, ((A/(2*B))-(qbar_j/2))) >= qbar_i && min(qbar_j, ((A/(2*B))-(qbar_i/2))) >= qbar_j
                mut.q_BR[i,j] = qbar_i

            #partial constrained case
            ###since I'm filling this in from the perspective of i, I don't need to worry about filling in the BR of j
            else
                mut.q_BR[i,j] = min(qbar_i, ((A/(2*B))-(qbar_j/2)))

            end #close the if statement 
            =#
        end #close loop over j
    end #close loop over i

    #we're not returing anything, but we know that the 

end #close function



=#