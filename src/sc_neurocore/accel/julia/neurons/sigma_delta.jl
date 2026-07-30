# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Sampled Yoon asynchronous pulse sigma-delta encoder."""
module SigmaDeltaAccel
export SigmaDeltaNeuronState, valid, step!, reset!, simulate
mutable struct SigmaDeltaNeuronState
    sigma::Float64; reconstruction::Float64; delta::Float64; tau_reconstruction::Float64; dt::Float64
end
SigmaDeltaNeuronState()=SigmaDeltaNeuronState(0.0,0.0,1.0,10.0,0.1)
valid(s::SigmaDeltaNeuronState)=all(isfinite,(s.sigma,s.reconstruction,s.delta,s.tau_reconstruction,s.dt))&&abs(s.sigma)<=1e12&&abs(s.reconstruction)<=1e12&&s.delta>0&&s.tau_reconstruction>0&&s.dt>0
"""Advance one atomic sampled APSDM transition."""
function step!(s::SigmaDeltaNeuronState,current::Float64=0.0)::Int
    (!isfinite(current)||!valid(s))&&throw(DomainError(current,"invalid SigmaDelta state or input"))
    sigma=s.sigma+s.dt*current; reconstruction=s.reconstruction*exp(-s.dt/s.tau_reconstruction); spike=sigma-reconstruction>=0.5*s.delta
    spike&&(reconstruction+=s.delta)
    (!isfinite(sigma)||!isfinite(reconstruction)||abs(sigma)>1e12||abs(reconstruction)>1e12)&&throw(DomainError((sigma,reconstruction),"SigmaDelta candidate outside safety envelope"))
    s.sigma=sigma;s.reconstruction=reconstruction;return spike ? 1 : 0
end
reset!(s::SigmaDeltaNeuronState)=(s.sigma=0.0;s.reconstruction=0.0;nothing)
function simulate(currents::AbstractVector{<:Real};state::SigmaDeltaNeuronState=SigmaDeltaNeuronState())
    sigma=Vector{Float64}(undef,length(currents));reconstruction=similar(sigma);events=Vector{Int64}(undef,length(currents))
    for i in eachindex(currents);events[i]=step!(state,Float64(currents[i]));sigma[i]=state.sigma;reconstruction[i]=state.reconstruction;end
    return (;sigma,reconstruction,events,state)
end
end
