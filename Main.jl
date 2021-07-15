# Cascade Correlation

using CSV
using Plots
using Random

# Hyperparameters
learning_rate_hid_in = 0.01 # learning rate for input-hidden weights (when adding hidden unit)
learning_rate_out = 0.01    # learning rate for hidden-output and input-output weights (for delta rule retraining)
eps_delta = 0.001           # precision for (re)training input-output and hidden-ouput weights
eps_cand = 0.5              # precision for adding hidden units (if after adding hidden unit error decrease is less then eps_cascade, stop adding)
max_iter_delta = 500        # max iterations for -output retraining
max_iter_cand = 500         # max iterations for candidate unit training (input-hidden_candidate)
n_candidates = 10           # how many candidate units will be initialized on adding each hidden neuron
target_hidden = 10          # target (maximum) amount of hidden units

# Activation functions for hidden and output units; activations for outputs are linear since we need to approximate function in real numbers
activation(x) = tanh.(x)
activation_der(x) = sech.(x).^2
activation_out(x) = x
activation_out_der(x) = 1.0

include("CascadeCorrelation.jl")
include("Plotting.jl")
include("ReadData.jl")
include("GenData.jl")

# Define a network model
mutable struct NN_model
    n_input :: Int              # input units
    n_hidden :: Int             # hidden units
    w_io :: Array{Float64, 2}   # input-output weights
    w :: Array{Float64, 2}      # input-hidden weights
    w_0 :: Vector{Float64}      # hidden bias
    w_hh :: Array{Float64, 2}   # hidden-hidden weights
    v :: Vector{Float64}        # hidden-output weights
    v_0 :: Real                 # output bias
end

# Define Julia's copy function for defined struct
Base.copy(model :: NN_model) = NN_model(
    model.n_input,
    model.n_hidden,
    model.w_io,
    model.w,
    model.w_0,
    model.w_hh,
    model.v,
    model.v_0
)

# Get some data
#(training_set_in, training_set_out) = read_data(filename_data)
#(training_set_in, training_set_out) = gen_data(100, "linear_bin")
#(training_set_in, training_set_out) = gen_data(300, "circular_bin")
#(training_set_in, training_set_out) = gen_data(300, "quintic_bin")
#(training_set_in, training_set_out) = gen_data(400, "sin_bin")
#(training_set_in, training_set_out) = gen_data(500, "spiral_bin")
#(training_set_in, training_set_out) = gen_data(500, "cubic3d")
(training_set_in, training_set_out) = gen_data(100, "parabola")

# Train Cascade Correlation Network
nn_model, err_arr =
    @time cascade_correlation(  training_set_in,
                                training_set_out,
                                learning_rate_hid_in,
                                learning_rate_out,
                                eps_delta,
                                eps_cand,
                                max_iter_delta,
                                max_iter_cand,
                                n_candidates,
                                target_hidden
    )

# Plot the results
#fig1 = plot_decision_boundary(nn_model, err_arr, training_set_in, training_set_out)
fig1 = plot_graph(nn_model, err_arr, training_set_in, training_set_out)
#fig1 = plot_graph_3d(nn_model, err_arr, training_set_in, training_set_out)
fig2 = plot_loss(err_arr)