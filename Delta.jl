# Delta Rule implementation

function delta( nn_model,
                training_set_in,
                training_set_out,
                learning_rate_out,
                eps_delta,
                max_iter_delta)

  println("Applying Delta Rule...")

  # Squared error between estimated and target output
  err = Inf
  err_prev = 0.0

  # Amount of examples
  n_examples = size(training_set_out, 1)

  # Current learning rate
  learning_rate_out_curr = learning_rate_out

  # Gradient descent iterations (with endless loop protection)
  for iter=1:max_iter_delta

    err_prev = err
    err = 0.0
    # Weighted sum of inputs of the output unit
    sum_y = 0.0
    # Decrease learning rate
    learning_rate_out_curr *= 0.99

    # Shuffle examples
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # Iterate through each training example
    for i=1:n_examples
      z, y, _, sum_y = feedforward(training_set_in[i,:], nn_model)

      # Stochastic gradient descent
      d_v_0 = zeros(size(nn_model.v_0))
      d_v = zeros(size(nn_model.v))
      d_w_io = zeros(size(nn_model.w_io))

      # ----- GRADIENT DESCENT -----
      # Calculating differences: output bias, in-out weights, hid-out weights
      d_v_0 = learning_rate_out_curr * (training_set_out[i] - y) * activation_out_der(sum_y) * 1
      d_w_io = learning_rate_out_curr * (training_set_out[i] - y) * activation_out_der(sum_y) .* training_set_in[i,:]
      if length(z) > 0
        d_v = learning_rate_out_curr * (training_set_out[i] - y) * activation_out_der(sum_y) .* z[:]
      end

      # Update weights (SGD)
      nn_model.v_0 += d_v_0
      nn_model.w_io += d_w_io'
      nn_model.v += d_v

      # Increment error between target and calculated output
      err += (training_set_out[i] - y) ^ 2
    end

    # Show current error
    (mod(iter, 100) == 0) ? (println("Iter:", iter, "; Error:", err, "; learning rate: ", learning_rate_out_curr)) : nothing

    # Check precision and stop if needed
    (abs(err - err_prev) < eps_delta) ? break : nothing

  end

  return (nn_model::NN_model, err)

end
