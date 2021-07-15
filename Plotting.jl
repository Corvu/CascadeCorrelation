# Plotting functions used in the example

# Plots decision boundary for given NN model
function plot_decision_boundary(nn_model, err_arr, training_set_in, training_set_out)
    gr()
    x_pl = -10.0:0.1:10
    y_pl = -10.0:0.1:10
    f(x,y) = begin
        feedforward([x,y], nn_model)[2]
    end
    X = repeat(reshape(x_pl, 1, :), length(y_pl), 1)
    Y = repeat(y_pl, 1, length(x_pl))
    Z = map(f, X, Y)

    scatter(training_set_in[training_set_out .== 1.0, 1],
        training_set_in[training_set_out .== 1.0, 2],
        marker=:x)
    scatter!(training_set_in[training_set_out .== -1.0, 1],
        training_set_in[training_set_out .== -1.0, 2],
        marker=:o)
    xlims!((-10, 10))
    ylims!((-10, 10))
    contour!(x_pl, y_pl, f, fill=false, levels=1)
end

function plot_graph(nn_model, err_arr, training_set_in, training_set_out)
    gr()
    x_pl = collect(-5.0:0.1:5)
    y_pred = collect(-5.0:0.1:5)
    for i = 1 : size(x_pl, 1)
        y_pred[i] = feedforward([x_pl[i]], nn_model)[2]
    end

    scatter(training_set_in[:, 1],
        training_set_out[:],
        marker=:o)
    
    xlims!((-5, 5))
    ylims!((0, 25))
    plot!(x_pl, y_pred)
end

function plot_graph_3d(nn_model, err_arr, training_set_in, training_set_out)
    pyplot()
    x_pl = -8.0:0.1:8
    y_pl = -8.0:0.1:8
    f_real(x,y) = begin
        x .^ 3 + y .^ 3
    end
    f(x,y) = begin
        feedforward([x,y], nn_model)[2]
    end
    X = repeat(reshape(x_pl, 1, :), length(y_pl), 1)
    Y = repeat(y_pl, 1, length(x_pl))
    Z_real = map(f_real, X, Y)
    Z = map(f, X, Y)

    #contour(x_pl, y_pl, f_real, fill=true, levels=10)
    surface(x_pl, y_pl, Z_real)
    xlims!(-10, 10)
    ylims!(-10, 10)
    zlims!(-1500, 1500)
    surface!(x_pl, y_pl, Z)
    xlims!(-10, 10)
    ylims!(-10, 10)
    zlims!(-1500, 1500)
end

# Plot data points
#= function plot_data(training_set_in, training_set_out)
    gr()
    scatter(training_set_in[training_set_out .== 1.0, 1],
        training_set_in[training_set_out .== 1.0, 2],
        marker=:x)
    scatter!(training_set_in[training_set_out .== -1.0, 1],
        training_set_in[training_set_out .== -1.0, 2],
        marker=:o)
    xlims!((-10, 10))
    ylims!((-10, 10))
end
=#

function plot_loss(err_arr)
    gr()
    plot(0:length(err_arr)-1, err_arr, title="Loss", xlabel="n_hidden", marker=:o)
end