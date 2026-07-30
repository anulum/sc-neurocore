# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Retained SC bipolar sigma-delta accumulator."""
module SCSigmaDeltaAccumulatorAccel
export SCSigmaDeltaAccumulatorState,valid,step!,simulate
mutable struct SCSigmaDeltaAccumulatorState;sigma::Float64;v_threshold::Float64;end
SCSigmaDeltaAccumulatorState()=SCSigmaDeltaAccumulatorState(0.0,1.0)
valid(s::SCSigmaDeltaAccumulatorState)=isfinite(s.sigma)&&isfinite(s.v_threshold)&&s.v_threshold>0
function step!(s::SCSigmaDeltaAccumulatorState,current::Float64)::Int
    (!isfinite(current)||!valid(s))&&throw(DomainError(current,"invalid SC SigmaDelta accumulator"));sigma=s.sigma+current;!isfinite(sigma)&&throw(DomainError(sigma,"non-finite candidate"));event=0
    if sigma>=s.v_threshold;sigma-=s.v_threshold;event=1;elseif sigma<=-s.v_threshold;sigma+=s.v_threshold;event=-1;end
    s.sigma=sigma;return event
end
function simulate(currents::AbstractVector{<:Real};state::SCSigmaDeltaAccumulatorState=SCSigmaDeltaAccumulatorState())
    sigma=Vector{Float64}(undef,length(currents));events=Vector{Int64}(undef,length(currents));for i in eachindex(currents);events[i]=step!(state,Float64(currents[i]));sigma[i]=state.sigma;end;return(;sigma,events,state)
end
end
