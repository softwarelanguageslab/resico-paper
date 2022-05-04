using CSV, DataFrames

function main()
    CSV_PATH = "/mansion/cavelazq/PhD/COSTER/data/contextsRESICO.csv"
    data = DataFrame(CSV.File(CSV_PATH))

    fqns = data.fqn
    unique_fqns = unique(fqns)

    mapping_fqn = map(enumerate(unique_fqns)) do (index, fqn) 
        fqn => index
    end
    
    mapping = Dict(mapping_fqn)

    open("mapping.txt", "w") do file
        for fqn in fqns
            write(file, "$(mapping[fqn]),$fqn\n")
        end
    end

    open("fqns_transformed.txt", "w") do file
        for fqn in fqns
            write(file, "$(mapping[fqn])\n")
        end
    end
end

main()
