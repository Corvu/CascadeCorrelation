# Plotting functions used in the example

# Plots decision boundary for given NN model
function plot_decision_boundary(nn_model, err_arr)
    gr()
    x_pl = 0.0:0.1:10
    y_pl = 0.0:0.1:10
    f(x,y) = begin
        feedforward([x,y], nn_model)[2]
    end
    X = repeat(reshape(x_pl, 1, :), length(y_pl), 1)
    Y = repeat(y_pl, 1, length(x_pl))
    Z = map(f, X, Y)
    contour!(x_pl, y_pl, f, fill=false)
end

# Plot data points
function plot_data(training_set_in, training_set_out)
    gr()
    scatter(training_set_in[training_set_out .== 1.0, 1],
        training_set_in[training_set_out .== 1.0, 2],
        marker=:+)
    scatter!(training_set_in[training_set_out .== -1.0, 1],
        training_set_in[training_set_out .== -1.0, 2],
        marker="x")
    xlims!((0, 10))
    ylims!((0, 10))
end
