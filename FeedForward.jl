# Feed-Forward of the CC Neural Network
# x - input units [n_examples, n_dims]
# nn_model - CCNN model

function feedforward(x::Array{Float64,1}, nn_model)

  # Calculated outputs of each hidden unit (with activation applied)
  z = zeros(nn_model.n_hidden)
  # Output of the network
  sum_h::Float64 = 0    # weighted sum of inputs of the last added hidden unit
  sum_y::Float64 = 0    # weighted sum of inputs of the output unit

  # --- CLACULATING Z - OUTPUTS OF HIDDEN UNITS ---
  # Iterate through hidden units
  for i=1:nn_model.n_hidden
    # Weighted sum of inputs for the hidden unit
    sum_h = nn_model.w_0[i] + sum(nn_model.w * x)
    # Iterate through preceding hidden units; hidden-hidden connections
    for j=1:i-1
      sum_h += nn_model.w_hh[i,j] * z[j]
    end
    # Output of the hidden unit with activation applied
    z[i] = activation(sum_h)
  end

  # --- CALCULATING Y - OUTPUT OF THE NETWORK ---
  # Weighted sum of inputs for the output unit and activation applied to it
  if (length(nn_model.v) > 0)
    sum_y = nn_model.v_0 + sum(nn_model.w_io * x) + sum(nn_model.v .* z)
  else
    sum_y = nn_model.v_0 + sum(nn_model.w_io * x)
  end
  y = activation(sum_y)

  return (z, y, sum_h, sum_y)
  # Returned tuple:
  # z - output values of hidden units
  # y - output value of network
  # sum_h - weighted sum of inputs of last added hidden unit
  # sum_y - weighted sum of inputs of output unit
end
