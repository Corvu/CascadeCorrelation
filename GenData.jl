# Generate data for testing

function gen_data(n_points, type)

    # Circular cluster of data
    if type == "circular_bin"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 16 .- 8
            if abs( (x_arr[i,1] - 1.0)^2 + (x_arr[i,2] - 1.0)^2 ) > 3^2
                y_arr[i] = -1.0
            else
                y_arr[i] = 1.0
            end
        end

    # Linear separation
    elseif type == "linear_bin"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 10
            if x_arr[i,2] > 8.0 - 0.8 * x_arr[i,1]
                y_arr[i] = -1.0
            else
                y_arr[i] = 1.0
            end
        end

    # Quintic function
    elseif type == "quintic_bin"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 16 .- 8
            if x_arr[i,2] > x_arr[i,1] ^ 5 - 3 * x_arr[i,1] ^ 3
                y_arr[i] = 1.0
            else
                y_arr[i] = -1.0
            end
        end

    elseif type == "sin_bin"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 16 .- 8
            if x_arr[i,2] > 4 * sin(x_arr[i,1])
                y_arr[i] = 1.0
            else
                y_arr[i] = -1.0
            end
        end
    
    elseif type == "spiral_bin"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        theta_arr = rand(n_points) * 3 * pi
        # Generation of each point
        for i=1:Int(floor(n_points/2))
            x_arr[i,1] = 1 * theta_arr[i] * cos(theta_arr[i])
            x_arr[i,2] = 1 * theta_arr[i] * sin(theta_arr[i])
            y_arr[i] = 1.0
        end
        for i=(Int(floor(n_points/2))+1):n_points
            x_arr[i,1] = 1 * theta_arr[i] * cos(theta_arr[i] + pi)
            x_arr[i,2] = 1 * theta_arr[i] * sin(theta_arr[i] + pi)
            y_arr[i] = -1.0
        end

    elseif type == "cubic3d"
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2) * 16 .- 8
            y_arr[i] = x_arr[i,1] ^ 3 + x_arr[i,2] ^ 3
        end
    
    elseif type == "parabola"
        # Arguments and labels
        x_arr = zeros(n_points,1)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,1] = rand() * 10.0 .- 5.0
            y_arr[i] = x_arr[i,1] ^ 2
        end

    end

    return x_arr, y_arr

end
