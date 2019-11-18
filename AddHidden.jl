# Add hidden unit to the network

function add_hidden(training_set_in,
					training_set_out,
					nn_model,
					n_candidates,
					learning_rate_hid_in)

	n_input = size(training_set_in,2)
	n_out = size(training_set_out,2)

	# Candidate units
	#w_cand = Array{Array{Float64, nn_model.n_input}, n_candidates} # input -> new_hidden
	w_cand = zeros(n_candidates, nn_model.n_input) # input -> new_hidden
	w_0_cand = zeros(n_candidates)	# new hidden bias
	#w_hh_cand = Array{Array{Float64, nn_model.n_hidden}, n_candidates} # hidden -> hidden; can only receive outputs of other units
	w_hh_cand = zeros(n_candidates, nn_model.n_hidden) # hidden -> hidden; can only receive outputs of other units

	# Weights of best candidate unit so far
	w_best_cand = zeros(nn_model.n_input)
	w_0_best_cand = 0.0
	w_hh_best_cand = zeros(nn_model.n_hidden)
	# Max correlation among candidates, will definetly be greater than zero after Adjusting inputs of hidden unit
	# TODO correlation or error???
	corr_max = 0.0

	# Create a pool of candidate units with different initial weights,
	# then optimize each as much as possible
	for c = 1:n_candidates

		print("Candidate unit #", c, "\n")

		w_cand[c,:] = transpose(rand(1,n_input)) * 0.1  # weights (input-hidden) [hidden_neuron,input_neuron]
		w_0_cand[c] = rand() * 0.1  # biases of each hidden neuron
		w_hh_cand[c,:] = transpose(rand(1,nn_model.n_hidden)) * 0.1 # weights (hidden-hidden) [hidden_neuron_from]

		# Correlation between output of hidden unit and residual output error of the network
		# (to decide which candidate unit is best)
		err_cand = 0.0

		# If no hidden units yet
		if (nn_model.n_hidden == 0)
			(w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], err_cand) =
			adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:])
		else
			(w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], err_cand) =
			adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:])
		end

		if (err_cand > corr_max)  # if this candidate is better
			print("Better candidate with error: ", err_cand, "\n")
			w_best_cand = w_cand[c,:]
			w_0_best_cand = w_0_cand[c]
			w_hh_best_cand = w_hh_cand[c,:]
			corr_max = err_cand
		end
	end

	nn_model.n_hidden += 1

	if (nn_model.n_hidden == 1)
		nn_model.w = transpose(w_best_cand)
		nn_model.w_0 = w_0_best_cand
		nn_model.w_hh = zeros(1,1)
		nn_model.v = rand(1,1) * 0.1
	else
		nn_model.w = [nn_model.w; transpose(w_best_cand)]
		nn_model.w_0 = [nn_model.w_0; w_0_best_cand]
		nn_model.w_hh = [nn_model.w_hh zeros(nn_model.n_hidden-1,1);
			transpose(w_hh_best_cand) 0.0]
		nn_model.v = [nn_model.v; rand() * 0.1]
	end

	return (nn_model :: NN_model, corr_max)

end
