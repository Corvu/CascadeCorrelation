# Read data from CSV, considering last column as labels and other as arguments

function read_data(filename_data)

    data_frame = CSV.read(filename_data)
    training_set_in::Array{Float64,2} = convert(Matrix, data_frame[:,1:size(data_frame)[2]-1])
    training_set_out::Array{Float64,1} = convert(Array, data_frame[:,size(data_frame)[2]])

    return (training_set_in, training_set_out)

end
