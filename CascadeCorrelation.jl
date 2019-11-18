#=
" Adjusting weights and adding new hidden neurons
# Arguments:

- `training_set_in`: [pattern, input]
- `training_set_out`: [pattern] (always 1 output)

# Return:

- `w_io` - weights from input to output units [output_neuron,input_neuron]
- `w` - input-hidden weights [hidden_neuron,input_neuron]
- `w_0` - bias of input-hidden weights [hidden_neuron]
- `w_hh` - hidden-hidden weights [hidden_neuron_to,hidden_neuron_from]
- `v` - hidden-output weights [output_neuron,hidden_neuron]
- `v_0` - bias of hidden-output weights [output_neuron]
"
=#

include("FeedForward.jl")
include("Delta.jl")
include("AdjustHidden.jl")
include("AddHidden.jl")
include("ShufflePatterns.jl")

function cascade_correlation( training_set_in::Array{Float64,2},
                              training_set_out::Array{Float64,1},
                              learning_rate_hid_in::Float64,
                              learning_rate_out::Float64,
                              eps_delta::Float64,
                              max_iter_delta::Int64)

  # Parameters and variables
  n_input = size(training_set_in, 2)
  n_examples = size(training_set_in, 1)

  # Initialize empty model with random weights
  nn_model = NN_model(n_input,
                      0,
                      rand(1, n_input) * 0.1,
                      zeros(n_input,0), # W_ih
                      0.0,  # w_h_0
                      zeros(0,0),  # W_hh
                      zeros(0,1),  # W_ho
                      rand() * 0.1) # y_0

  # Loss history
  err_arr = 0.0

  # Adjust input-output weights by Delta Rule as much as possible
  print("start n_hid: ", nn_model.n_hidden)
  (nn_model, err_init) =
    delta(nn_model, training_set_in, training_set_out, learning_rate_out, eps_delta, max_iter_delta)
  print("\nHidden units: 0", "\tError:", err_init, "\n")

  # If error is low enough already, don't add hidden units and return linear model
  if (abs(err_init) < eps_cascade)
    return (nn_model, err_arr)
  end

  # --- ADDING HIDDEN UNITS ---
  # Add first neuron into the Network (initialize several candidate units, train them, then choose the best one)
  nn_model.n_hidden = 0

  # Weights and biases (input-hidden) [hidden_neuron,input_neuron]
  nn_model.w = zeros(0, n_input)
  nn_model.w_0 = zeros(0)
  # Weights and biases (hidden-hidden) [hidden_neuron_to,hidden_neuron_from]
  nn_model.w_hh = 0.0
  nn_model.v = rand(1,1) * 0.1

  # TODO calculated values at the outputs of each hidden neuron
  z = zeros(nn_model.n_hidden)

  # Error tracking
  err_prev = 0.0
  err = Inf
  err_arr = zeros(max_hidden) # error of prediction for every amount of hidden units

  # Calculating prediction error and adding another hidden unit if needed
  for iteration = 1:max_hidden

    # Incremental squared error (to decide if we need another hidden unit)
    err_prev = err
    err = 0.0

    # Add hidden unit
    nn_model =
      add_hidden(training_set_in, training_set_out, nn_model, n_candidates, learning_rate_hid_in)[1]

    # Retrain input-output and hidden-output weights using delta rule
    (nn_model, err) =
      delta(nn_model, training_set_in, training_set_out, learning_rate_out, eps_delta, max_iter_delta)

    # Calculate error by summating differences between feed-forward results and training set
    for p = 1:size(training_set_in, 1)
      y = feedforward(training_set_in[p,:], nn_model)[2]
      err += (training_set_out[p] - y) ^ 2
    end

    # History of errors for each step of adding hidden units
    err_arr[iteration] = err
    print("\nHidden units:", iteration, "\tError:", err, "\n")

    # If error is low enough, stop adding hidden units
    if (abs(err - err_prev) < eps_cascade)
      break
    end

  end

  print("\nCascade Correlation training completed\n")
  print("Hidden units: ", nn_model.n_hidden, "\n\n")

  return (nn_model, err_arr)

end
