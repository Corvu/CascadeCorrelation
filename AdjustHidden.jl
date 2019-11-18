# Adding hidden neuron
# Applying gradient ascent on the input weights of the candidate hidden unit,
# input weights of chosen unit will be frozen after adding
# n_hidden - number of expected hidden units; one of them not connected yet

function adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in,
  w_cand_concr, w_0_cand_concr, w_hh_cand_concr)

  eps = 1 # patience for adjusting input weights of the candidate
  err_prev = 0.0
  err = Inf # incremental error (correlation) for the candidate hidden unit

  # Gradient ascent iterations; endless loop protection
  for iter=1:max_iter_cand

    err_prev = err
    err = 0.0
    # Decrease learning rate
    learning_rate_hid_in *= 0.98

    # Shuffle patterns
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # --- Calculating error after adding this hidden unit ---
    # Output of the new hidden unit (not yet connected) and output of the network,
    # both averaged over all training examples
    z_avg = 0.0
    e_avg = 0.0
    # Hidden unit output values and outut values of the network, both after feed-forward
    z_pattern = zeros(n_examples)
    y_pattern = zeros(n_examples)
    # Calculated weighted sum of inputs of last added hidden neuron, for each training example
    summ = zeros(n_examples)
    # Calculating averages for hidden unit output values and error in the network output
    for i=1:1 # for each hidden unit (?) TODO
      for j=1:size(training_set_in,1) # for each training pattern
        # Network output for each training example
        y_pattern[j] = feedforward(training_set_in[j,:], nn_model)[2]
        e_avg += (training_set_out[j] - y_pattern[j])	# not squared? TODO
        # Output of new hidden unit, for one training example
        z_pattern[j] = (transpose(w_cand_concr) * training_set_in[j,:])[1]
        z_avg += z_pattern[j]
      end
      z_avg = z_avg / size(training_set_in, 1)  # average output value of the hidden unit
      e_avg = e_avg / size(training_set_in, 1)  # average residual error

      # Iterate through training examples
      for j=1:size(training_set_in,1)
        # Calculate cumulative correlation for each amount of hidden units
        # The goal is to choose candidate unit with maximum correlation
        err += abs( (z_pattern[j] - z_avg) * (y_pattern[j] - e_avg) )
      end
    end

    # TODO Check for errors & delete
    # In case of gradient ascent, correlation decreases, but should increase
    if (mod(iter, 20) == 0)
      print("Iter: ", iter, " Error (corr?): ", err, "\n")
    end

    # Apply gradient ascent (optimizing input weights of the new hidden neuron (NHN))

    # Sign of the correlation between NHN output and residual output error
    sign_corr(a,b) = (a * b) / abs(a * b)

    # --- Calculate gradients for weights ---
    # Input-hidden weights bias of NHN
    d_w_0_cand_concr = 0.0
    for i=1:size(training_set_in,1)
      d_w_0_cand_concr += sign_corr(z_avg, e_avg) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(summ[i]) * 1 * learning_rate_hid_in
    end
    
    w_0_cand_concr += d_w_0_cand_concr

    # Input-hidden weights of NHN
    for j=1:n_input
      d_w_cand_concr = 0.0
      for i=1:size(training_set_in,1)
        d_w_cand_concr += sign_corr(z_avg, e_avg) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(summ[i]) * training_set_in[i,j] * learning_rate_hid_in
      end
      w_cand_concr[j] += d_w_cand_concr
    end

    # Hidden-hidden weights of NHN
    # TODO check for errors
    for j=1:nn_model.n_hidden
      d_b_cand_concr = 0.0
      for i=1:size(training_set_in,1)
        d_b_cand_concr += sign_corr(z_avg, e_avg) * ((training_set_out[i] - y_pattern[i]) - e_avg) * activation_der(summ[i]) * z_pattern[i] * learning_rate_hid_in
      end
      w_hh_cand_concr[j] += d_b_cand_concr
    end

    # If error (correlation) not improving, stop
    if (abs(err - err_prev) < eps)
      break
    end

  end

  return (w_cand_concr, w_0_cand_concr, w_hh_cand_concr, err)

end
