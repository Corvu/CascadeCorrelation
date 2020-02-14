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
                      rand(1, n_input) * 2 .- 1, # w_io
                      zeros(n_input,0), # W_ih
                      zeros(0),  # w_h_0
                      zeros(0,0),  # W_hh
                      zeros(0),  # W_ho
                      rand() * 2 - 1) # v_0

  # Loss history
  err_arr = 0.0

  # Adjust input-output weights by Delta Rule as much as possible
  println("start n_hid: ", nn_model.n_hidden)
  (nn_model, err_init) =
    delta(nn_model, training_set_in, training_set_out, learning_rate_out, eps_delta, max_iter_delta)
  println("Hidden units: 0", "\tError:", err_init)

  # If error is low enough already, don't add hidden units and return linear model
  if (abs(err_init) < eps_cascade)
    println("Achieved desired precision using delta rule; no hidden units added")
    return (nn_model, err_init)
  end

  # --- ADDING HIDDEN UNITS ---

  # Weights and biases (input-hidden) (hidden-hidden)
  nn_model.w = rand(0, n_input)
  nn_model.w_0 = rand(0)
  nn_model.w_hh = zeros(1,1)
  nn_model.v = rand(1)

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

    # History of errors for each step of adding hidden units
    err_arr[iteration] = err
    println("Hidden units:", iteration, "\tError:", err, "\n")

    # If error is low enough, stop adding hidden units
    if (abs(err - err_prev) < eps_cascade)
      break
    end

  end

  println("Cascade Correlation training completed")
  println("Hidden units: ", nn_model.n_hidden, "\n")

  return (nn_model, [err_init; err_arr])

end
