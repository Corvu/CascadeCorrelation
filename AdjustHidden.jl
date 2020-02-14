# Adding hidden neuron
# Applying gradient ascent on the input weights of the candidate hidden unit,
# input weights of chosen unit will be frozen after adding
# n_hidden - number of expected hidden units; one of them not connected yet

function adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in,
  w_cand_concr, w_0_cand_concr, w_hh_cand_concr)

  eps_hidden = 0.5 # patience for adjusting input weights of the candidate
  corr_prev = 0.0
  corr = Inf # incremental correlation for the candidate hidden unit

  # New candidate unit weights
  w_cand_concr_new = w_cand_concr
  w_0_cand_concr_new = w_0_cand_concr
  w_hh_cand_concr_new = w_hh_cand_concr

  # Set initial value of dynamic learning rate
  learning_rate_hid_in_curr = learning_rate_hid_in

  # Gradient ascent iterations
  for iter=1:max_iter_cand

    corr_prev = abs(corr)
    corr = 0.0
    # Decrease learning rate
    learning_rate_hid_in_curr *= 0.99

    # Shuffle patterns
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # --- Calculating correlation after adding this hidden unit ---
    # Output of the new hidden unit (not yet connected) and output of the network,
    # both averaged over all training examples
    z_avg = 0.0
    e_avg = 0.0
    # Values of candidate units and network outputs
    z_pattern = zeros(size(training_set_in, 1))
    y_pattern = zeros(size(training_set_in, 1))
    # Calculated weighted sum of inputs of last added hidden neuron, for each training example
    h_inp_pattern = zeros(size(training_set_in, 1))
    # Calculating averages for hidden unit output values and error in the network output
    for i=1:1 # for each output unit
      for j=1:size(training_set_in,1) # for each training pattern
        # Network output for each training example
        h_pattern, y_pattern[j], h_inp_pattern[j] = feedforward(training_set_in[j,:], nn_model)[1:3]
        e_avg += (training_set_out[j] - y_pattern[j])
        # Output of new hidden unit, for one training example
        z_pattern[j] = activation(w_0_cand_concr + (w_cand_concr' * training_set_in[j,:]) + (w_hh_cand_concr' * h_pattern))
        z_avg += z_pattern[j]
      end
      z_avg = z_avg / size(training_set_in, 1)  # average output value of the hidden unit
      e_avg = e_avg / size(training_set_in, 1)  # average residual error

      # Calculate cumulative correlation for each amount of hidden units
      # The goal is to choose candidate unit with maximum correlation
      for j=1:size(training_set_in,1)
        corr += (z_pattern[j] - z_avg) * ( (training_set_out[j] - y_pattern[j]) - e_avg)
      end
    end

    if (mod(iter, 50) == 0)
      println("Iter: ", iter, " Corr (signed): ", corr)
    end

    # Apply gradient ascent (optimizing input weights of the new hidden neuron (NHN))

    # --- Calculate gradients for weights ---
    # Input-hidden weights bias of NHN
    d_w_0_cand_concr = 0.0
    for i=1:size(training_set_in, 1)
      h_pattern = feedforward(training_set_in[i,:], nn_model)[1]
      d_w_0_cand_concr += sign(corr) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(w_0_cand_concr + w_cand_concr' * training_set_in[i,:] + w_hh_cand_concr' * h_pattern) * 1 * learning_rate_hid_in_curr
    end
    w_0_cand_concr_new += d_w_0_cand_concr

    # Input-hidden weights of NHN
    for j=1:size(training_set_in, 2)
      d_w_cand_concr = 0.0
      for i=1:size(training_set_in, 1)
        h_pattern = feedforward(training_set_in[i,:], nn_model)[1]
        d_w_cand_concr += sign(corr) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(w_0_cand_concr + w_cand_concr' * training_set_in[i,:] + w_hh_cand_concr' * h_pattern) * training_set_in[i,j] * learning_rate_hid_in_curr
      end
      w_cand_concr_new[j] += d_w_cand_concr
    end

    # Hidden-hidden weights of NHN
    for j=1:nn_model.n_hidden-1
      d_b_cand_concr = 0.0
      for i=1:size(training_set_in, 1)
        h_pattern = feedforward(training_set_in[i,:], nn_model)[1]
        d_b_cand_concr += sign(corr) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(w_0_cand_concr + w_cand_concr' * training_set_in[i,:] + w_hh_cand_concr' * h_pattern) * z_pattern[i] * learning_rate_hid_in_curr
      end
      w_hh_cand_concr_new[j] += d_b_cand_concr
    end

    # Update candidate unit weights
    w_0_cand_concr = w_0_cand_concr_new
    w_cand_concr = w_cand_concr_new
    w_hh_cand_concr = w_hh_cand_concr_new

    # If correlation not improving, stop
    if (abs(abs(corr) - corr_prev) < eps_hidden)
      break
    end

  end

  return (w_cand_concr, w_0_cand_concr, w_hh_cand_concr, abs(corr))

end
