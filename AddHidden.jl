# Add hidden unit to the network

function add_hidden(training_set_in,
					training_set_out,
					nn_model,
					n_candidates,
					learning_rate_hid_in)

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
				adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:])
		else
			(w_cand[c,:], w_0_cand[c], w_hh_cand[c,:], corr_cand) =
				adjust_hidden(nn_model, training_set_in, training_set_out, learning_rate_hid_in, w_cand[c,:], w_0_cand[c], w_hh_cand[c,:])
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
