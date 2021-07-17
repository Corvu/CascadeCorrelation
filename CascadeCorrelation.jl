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
include("ShufflePatterns.jl")

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


# Adding hidden neuron
# Applying gradient ascent on the input weights of the candidate hidden unit,
# input weights of chosen unit will be frozen after adding
# n_hidden - number of expected hidden units; one of them not connected yet
function adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in,
    w_cand_concr, w_0_cand_concr, w_hh_cand_concr, eps_cand, max_iter_cand)

    #eps_cand = 0.5   # patience for adjusting input weights of the candidate
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
        if (abs(abs(corr) - corr_prev) < eps_cand)
            break
        end

    end

    return (w_cand_concr, w_0_cand_concr, w_hh_cand_concr, abs(corr))

end


# Add hidden unit to the network
function add_hidden(training_set_in,
                    training_set_out,
                    nn_model,
                    n_candidates,
                    learning_rate_hid_in,
                    eps_cand,
                    max_iter_cand)

    n_input = size(training_set_in, 2)
    n_out = size(training_set_out, 2)

    # Candidate units
    w_cand = rand(n_candidates, nn_model.n_input) * 4 .- 2 		# input -> new_hidden
    w_0_cand = rand(n_candidates) * 4 .- 2						# new hidden bias
    w_hh_cand = rand(n_candidates, nn_model.n_hidden) * 4 .- 2 	# hidden -> hidden; can only receive outputs of other units

    # Weights of best candidate unit so far
    w_best_cand = zeros(nn_model.n_input)
    w_0_best_cand = 0.0
    w_hh_best_cand = zeros(nn_model.n_hidden)

    # Max correlation among candidates, will definetly be greater than zero after Adjusting inputs of hidden unit
    corr_max = 0.0

    # Tierate through candidate units
    for c = 1:n_candidates

        println("Candidate unit #", c)

        # Correlation between output of hidden unit and residual output error of the network (to decide which candidate unit is best)
        corr_cand = 0.0

        # If no hidden units yet
        if (nn_model.n_hidden == 0)
            (w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], corr_cand) =
                adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], eps_cand, max_iter_cand)
        else
            (w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], corr_cand) =
                adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], eps_cand, max_iter_cand)
        end

        # Check if this candidate is better
        if (corr_cand > corr_max)
            println("Better candidate with correlation: ", corr_cand)
            w_best_cand = w_cand[c,:]
            w_0_best_cand = w_0_cand[c]
            w_hh_best_cand = w_hh_cand[c,:]
            corr_max = corr_cand
        end
    end

    # Update values of existing model
    nn_model.n_hidden += 1
    if (nn_model.n_hidden == 1)
        nn_model.w = w_best_cand'
        nn_model.w_0 = [w_0_best_cand]
        nn_model.w_hh = zeros(1,1)
        nn_model.v = rand(1) * 4 .- 2
    else
        nn_model.w = [nn_model.w; w_best_cand']
        nn_model.w_0 = [nn_model.w_0; w_0_best_cand]
        nn_model.w_hh = [	nn_model.w_hh 		zeros(nn_model.n_hidden-1, 1);
                            w_hh_best_cand' 	0.0 ]
        nn_model.v = [nn_model.v; rand() * 4 .- 2]
    end

    return (nn_model :: NN_model, corr_max)

end


# Core algorithm
function cascade_correlation(   training_set_in::Array{Float64,2},
                                training_set_out::Array{Float64,1},
                                learning_rate_hid_in::Float64,
                                learning_rate_out::Float64,
                                eps_delta::Float64,
                                eps_cand::Float64,
                                max_iter_delta::Int64,
                                max_iter_cand::Int64,
                                n_candidates::Int64,
                                target_hidden::Int64,
                                nn_model)

    # Parameters and variables
    n_input = size(training_set_in, 2)
    n_examples = size(training_set_in, 1)

    # If model is empty, initialize new one with random weights (in-out, out_bias) and no hidden units
    if (nn_model === nothing)
        nn_model = NN_model(n_input,                    # n_input
                            0,                          # n_hidden
                            rand(1, n_input) * 10 .- 5, # w_io
                            zeros(n_input, 0),          # W_ih
                            zeros(0),                   # w_h_0
                            zeros(0,0),                 # W_hh
                            zeros(0),                   # W_ho
                            rand() * 10 - 5)            # v_0
    end
    
    # Loss history
    err_arr = 0.0

    # Adjust input-output weights by Delta Rule as much as possible
    println("start n_hid: ", nn_model.n_hidden)
    (nn_model, err_init) =
        delta(nn_model, training_set_in, training_set_out, learning_rate_out, eps_delta, max_iter_delta)
    println("Hidden units: 0", "\tError:", err_init)

    # If error is low enough already, don't add hidden units and return linear model
    if (abs(err_init) < eps_cand)
        println("Achieved desired precision using delta rule; no hidden units added")
        return (nn_model, err_init)
    end

    # If max amount of hidden units was reached, return
    if nn_model.n_hidden >= target_hidden
        return (nn_model, err_init)
    end

    # --- ADDING HIDDEN UNITS ---

    # If model is empty, initialize Weights and biases for everything related to hidden units (in-hid, hid_bias, hid-hid, hid-out)
    if (nn_model === nothing)
        nn_model.w = rand(0, n_input)
        nn_model.w_0 = rand(0)
        nn_model.w_hh = zeros(1,1)
        nn_model.v = rand(1)
    end

    # Remember to keep history of prediction error for every amount of hidden units
    err_prev = 0.0
    err = Inf
    err_arr = zeros(target_hidden-nn_model.n_hidden)

    # Adding hidden units until precision is satisfied or until target number of units was reached
    for iteration = 1:target_hidden-nn_model.n_hidden

        # Incremental squared error (to decide if we need another hidden unit)
        err_prev = err
        err = 0.0

        # Add hidden unit
        nn_model =
            add_hidden(training_set_in, training_set_out, nn_model, n_candidates, learning_rate_hid_in, eps_cand, max_iter_cand)[1]

        # Retrain input-output and hidden-output weights using delta rule
        (nn_model, err) =
            delta(nn_model, training_set_in, training_set_out, learning_rate_out, eps_delta, max_iter_delta)

        # History of errors for each step of adding hidden units
        println(err_arr)
        err_arr[iteration] = err
        println("Hidden units:", iteration, "\tError:", err, "\n")

        # If error is low enough, stop adding hidden units
        if (abs(err - err_prev) < eps_cand)
            break
        end

    end

    println("Cascade Correlation training completed")
    println("Hidden units: ", nn_model.n_hidden, "\n")

    return (nn_model, [err_init; err_arr])

end
