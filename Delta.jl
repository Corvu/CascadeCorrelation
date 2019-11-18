# Delta Rule implementation

function delta( nn_model,
                training_set_in,
                training_set_out,
                learning_rate_out,
                eps_delta,
                max_iter_delta)

  print("\nApplying Delta Rule...\n")

  # Squared error between y and y_target
  err = Inf
  err_prev = 0.0

  # Amount of examples
  n_examples = size(training_set_out, 1)

  # Gradient descent iterations (with endless loop protection)
  for iter=1:max_iter_delta

    err_prev = err
    err = 0.0
    # Weighted sum of inputs of the output unit
    sum_y = 0.0
    # Decrease learning rate
    learning_rate_out *= 0.98

    # Shuffle examples
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # Iterate through each training example
    for i=1:n_examples
      z, y, _, sum_y = feedforward(training_set_in[i,:], nn_model)

      # Difference between target output value and calculated one
      #e_out = training_set_out[i] - y

      # Stochastic gradient descent (ascent)
      d_v_0 = zeros(size(nn_model.v_0))
      d_v = zeros(size(nn_model.v))
      d_w_io = zeros(size(nn_model.w_io))

      # ----- GRADIENT DESCENT -----
      # Calculating differences
      # Output bias
      d_v_0 = learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) * 1
      # Input-output weights
      d_w_io = learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) .* training_set_in[i,:]
      # Hidden-output weights
      if length(z) > 0
        d_v = learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) .* z[:]
      end

      # Update weights (SGD)
      nn_model.v_0 += d_v_0
      nn_model.w_io += d_w_io'
      nn_model.v += d_v
      nn_model.v_0 = nn_model.v_0[1]

      # Increment error between target and calculated output
      err += (training_set_out[i] - y)^2
    end

    # Show current error
    (mod(iter, 100) == 0) ? (print("Iter:", iter, "; Error:", err, "\n")) : nothing

    # Check precision and break if needed
    (abs(err - err_prev) < eps_delta) ? break : nothing

  end

  return (nn_model::NN_model, err)

end
