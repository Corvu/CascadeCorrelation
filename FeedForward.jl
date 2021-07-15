# Feed-Forward of the CC Neural Network
# x - input units [n_dims]
# nn_model - CCNN model

function feedforward(x :: Vector{Float64}, nn_model)

    # Calculated outputs of each hidden unit (with activation applied)
    z = zeros(nn_model.n_hidden)
    # Output of the network
    sum_h :: Float64 = 0.0    # weighted sum of inputs of the last added hidden unit
    sum_y :: Float64 = 0.0    # weighted sum of inputs of the output unit

    # --- CLACULATING Z - OUTPUTS OF HIDDEN UNITS ---
    # Iterate through hidden units
    for i=1:nn_model.n_hidden
        # Weighted sum of inputs for the hidden unit
        sum_h = nn_model.w_0[i] + nn_model.w[i,:]' * x + nn_model.w_hh[i,:]' * z
        # Iterate through preceding hidden units; hidden-hidden connections
        #for j=1:i-1
        #  sum_h += nn_model.w_hh[i,j] * z[j]
        #end
        # Output of the hidden unit with activation applied
        z[i] = activation(sum_h)
    end

    # --- CALCULATING Y - OUTPUT OF THE NETWORK ---
    # Weighted sum of inputs for the output unit and activation applied to it
    if (nn_model.n_hidden > 0)
        sum_y = nn_model.v_0 + nn_model.w_io[:]' * x + nn_model.v' * z
    else
        sum_y = nn_model.v_0 + nn_model.w_io[:]' * x
    end
    y = activation_out(sum_y)

    return (z, y, sum_h, sum_y)
    # Returned tuple:
    # z - output values of hidden units
    # y - output value of network
    # sum_h - weighted sum of inputs of last added hidden unit
    # sum_y - weighted sum of inputs of output unit
end
