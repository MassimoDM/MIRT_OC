# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-08T12:46:39+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: helper_functions.jl
# @Last modified by:   massimo
# @Last modified time: 2021-01-25T20:18:36+01:00
# @License: LGPL-3.0
# @Copyright: {{copyright}}

using OpenBB
using SparseArrays





function load_constraint_set(pack::Dict{Any,Any},libID::Int=-1)::Tuple{Int,Dict{String,Any}}

    # collect info
    numVars,numCnss,lenPars = pack["sizes"]
    numPars = length(pack["param_vector"])

    # store a local copy of the parameters for the functions and of the constraints bounds
    paramVector = copy(pack["param_vector"])
    loBs = copy(pack["lob"])
    upBs = copy(pack["upb"])

    # load the necessary lib in OpenBB
    if libID == -1
        libID = load_lib(pack["lib_path"])
    end

    # share shorthand definitions
    @everywhere begin
        const Float = Float64
        const SpVector = OpenBB.SpVector
        const SpMatrix = OpenBB.SpMatrix
    end


    # create the evaluation function
    @everywhere begin
        function evalVal(x::Vector{Float})::Vector{Float}
            out = Vector{Float}(undef,$numCnss)
            (ccall(OpenBB.dlsym(OpenBB.get_lib_pointer($libID),:simplified_eval),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,$paramVector,out))
            return out
        end
    end

    # create jacobian
    (rowsJ,colsJ) = (pack["jcb_sparsity"][1].+1,pack["jcb_sparsity"][2].+1)
    nnzJ = pack["jcb_nnz"]
    jcbSparsity = Dict("rows"=>rowsJ,"cols"=>colsJ,"vals"=>true,"n"=>numCnss,"m"=>numVars)
    @everywhere begin
        function evalJcb(x::Vector{Float})::SpMatrix{Float}
            values = Vector{Float}(undef,$nnzJ)
            ccall(OpenBB.dlsym(OpenBB.get_lib_pointer($libID),:simplified_eval_jac),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,$paramVector,values)
            return OpenBB.sparse($rowsJ,$colsJ,values,$numCnss,$numVars)
        end
    end

    # create hessians
    rowsH = [pack["hes_sparsity"][k][1].+1 for k in 1:numCnss]
    colsH = [pack["hes_sparsity"][k][2].+1 for k in 1:numCnss]
    nnzH = copy(pack["hes_nnz"])
    hesSparsity = [Dict("rows"=>rowsH[k],"cols"=>colsH[k],"vals"=>true,"n"=>numVars,"m"=>numVars) for k in 1:numCnss]
    @everywhere begin
        function evalHes(x::Vector{Float})::Vector{SpMatrix{Float}}
            out = Vector{SpMatrix{Float}}(undef,$numCnss)
            for k in 1:$numCnss
                if $nnzH[k] > 0
                    values = Vector{Float}(undef,$nnzH[k])
                    ccall(OpenBB.dlsym(OpenBB.get_lib_pointer($libID),Symbol(:simplified_eval_hes,k-1)),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,$paramVector,values)
                    out[k] = OpenBB.sparse($rowsH[k],$colsH[k],values,$numVars,$numVars)
                else
                    out[k] = OpenBB.sparse(Int[],Int[],Float[],$numVars,$numVars)
                end
            end
            return out
        end
    end
    
    return  (libID,
             Dict{String,Any}("type"=>"Convex",
                              "evalVal"=>Main.evalVal,"evalJcb"=>Main.evalJcb,"evalHes"=>Main.evalHes,
                              "typeJcb"=>SpMatrix{Float},"typeHes"=>SpMatrix{Float},
                              "jcbSparsity"=>jcbSparsity,"hesSparsity"=>hesSparsity,
                              "loBs"=>loBs,"upBs"=>upBs))
end


function load_objective_fun(pack::Dict{Any,Any},libID::Int=-1)::Tuple{Int,Dict{String,Any}}

    # collect info
    numVars,~,lenPars = pack["sizes"]
    numPars = length(pack["param_vector"])

    # store a local copy of the parameters for the functions
    paramVector = copy(pack["param_vector"])

    # load the necessary lib in OpenBB
    if libID == -1
        libID = load_lib(pack["lib_path"])
    end

    # create the evaluation function
    function evalVal(x::Vector{Float})::Float
        out = Vector{Float}(undef,1)
        (ccall(dlsym(get_lib_pointer(libID),:simplified_eval),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,paramVector,out))
        return out[1]
    end

    # create gradient
    indsG = pack["grd_sparsity"].+1
    nnzG = pack["grd_nnz"]
    grdSparsity = Dict("inds"=>indsG,"vals"=>true,"n"=>numVars)
    function evalGrd(x::Vector{Float})::SpVector{Float}
        values = Vector{Float}(undef,nnzG)
        ccall(dlsym(get_lib_pointer(libID),:simplified_eval_jac),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,paramVector,values)
        return sparsevec(indsG,values,numVars)
    end

    # create hessian
    (rowsH,colsH) = (pack["hes_sparsity"][1].+1,pack["hes_sparsity"][2].+1)
    nnzH = pack["hes_nnz"]
    hesSparsity = Dict("rows"=>rowsH,"cols"=>colsH,"vals"=>true,"n"=>numVars,"m"=>numVars)
    function evalHes(x::Vector{Float})::SpMatrix{Float}
        if nnzH > 0
            values = Vector{Float}(undef,nnzH)
            ccall(dlsym(get_lib_pointer(libID),:simplified_eval_hes0),Cint,(Ptr{Float},Ptr{Float},Ptr{Float}),x,paramVector,values)
            return sparse(rowsH,colsH,values,numVars,numVars)
        else
            return sparse(Int[],Int[],Float[],numVars,numVars)
        end
    end


    return (libID,
            Dict{String,Any}("type"=>"Convex",
                             "evalVal"=>evalVal,"evalGrd"=>evalGrd,"evalHes"=>evalHes,
                             "typeGrd"=>SpVector{Float},"typeHes"=>SpMatrix{Float},
                             "grdSparsity"=>grdSparsity,"hesSparsity"=>hesSparsity))
end
