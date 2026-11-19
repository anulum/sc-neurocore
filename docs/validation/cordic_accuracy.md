<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Circular CORDIC fixed-point accuracy
-->

# Circular CORDIC fixed-point accuracy

The [clocked circular unit](../compiler/standalone_function_units.md#clocked-circular-cordic)
operates on signed fixed-point codes. Let $F$ be the fractional width, $h = 2^{-(F+10)}$
the internal quantum and $N = F+11 \le 41$ the iteration count. The ideal reference is
the real-valued function of those exact input codes, clipped to the signed output
interval. The final output differs from that reference by at most $11/16$ of an
output LSB. It differs by at most one code from the ideal reference rounded
half up and saturated to the same word. Here half-up rounding adds half a
quantum before flooring, so signed ties round toward positive infinity.

The circular shift-and-add recurrence is the rotation and vectoring method of
[Volder](https://www.ece.ucdavis.edu/~vojin/CLASSES/EPFL/Papers/3-Volder_CORDIC.pdf).
The angle of the $i$th exact micro-rotation is $a_i = \arctan(2^{-i})$ and
its gain is $\sqrt{1+2^{-2i}}$. [Walther](https://www.ece.ucdavis.edu/~vojin/CLASSES/EPFL/Papers/21-Welther-El_funct.pdf)
gives the circular recurrence and its convergence domain.

## Encoded constants and word bounds

Angles and the inverse gain are rounded half up in units of $h$. For all
31 supported fractional widths, rational interval enclosures verify each
stored angle, the internal and wide pi values, and the inverse gain. The
arctangent intervals use the first omitted term of the exact alternating
series. Machin's formula gives the interval for pi. The inverse gain is
checked against the exact product of the squared gains. The product of
the gains is below $1.647$ for every used sequence.

The unsigned remainder is below two pi in its own fixed-point representation
at each reduction edge. Doubling it before conditional subtraction still
fits the double-remainder register. Coordinate scaling starts from magnitudes
at most $2^{W-1}$ in integer codes and shifts until at least one coordinate
reaches $2^{F+10}$, where $W$ is the input word width. The magnitude gain is
less than $1.647$. The coordinate register width of
$\max(W+3,F+14)$ accommodates the resulting coordinates and shift errors.
The signed angle and final rounding addition fit the same width.

## Rotation error

The wide reduction rounds pi at $F+44$ fractional bits. A full signed input
requires at most $2^{31}+1$ period subtractions. Period quantisation,
rounding to the internal fraction and the subsequent quadrant fold together
perturb the intended rotation by less than $2h$, taking angles modulo a
full turn.

Let $A_i$ be the integer code for $a_i$ and $z_i$ the rotation residual
in internal codes. The folded phase satisfies $|z_0| \le 2A_0+1$.
The rounded angle codes satisfy $A_i \le 2A_{i+1}+1$ and $A_{N-1}=1$.
The update subtracts $A_i$ from the residual magnitude before taking the
absolute value. By induction, $|z_i| \le 2A_i+i+1$ before each step and
$|z_N| \le N+1$ afterwards. Adding the table rounding error gives a
total angular discrepancy of at most $(N+1+N/2+2)h$.

The two arithmetic shifts add at most $\sqrt{2}h$ in coordinate error per
step. The remaining exact gains amplify that error by at most $1.647$.
The initial inverse gain is within $h/2$ of its exact value. Each output
coordinate therefore has error bounded by

$$
\left[1.647\left(\tfrac12+N\sqrt{2}\right)
       +N+1+\tfrac N2+2\right]h < 192h.
$$

## Vectoring error

For nonaxial inputs the shared scale makes the initial norm at least one
in internal units. After any of the at most 41 steps the norm is at least
$1-41\sqrt{2}h$. The first coordinate stays positive under the signed
vectoring update. Each exact rotation followed by coordinate shift
quantisation perturbs the vector's angle by at most $2h$, using the norm
lower bound to bound the angular change of an error vector of length
$\sqrt{2}h$.

The initial first-quadrant argument is at most $2a_0$. The inequality
$a_i \le 2a_{i+1}$ and steering by the sign of the current second coordinate
give a residual angle at most $a_{N-1}+2Nh \le (1+2N)h$. Early termination
with the second coordinate zero gives zero residual. Summing the residual,
per-step angle perturbation, angle-table quantisation and quadrant
reconstruction gives at most $(1.5+4.5N)h < 192h$.
The axial branches use the encoded exact axial angles directly.
Univariate arctangent uses vectoring with a positive unit horizontal
coordinate.

## Output rounding

Half-up rounding adds at most half an output LSB. Signed saturation is
nonexpansive. Therefore the final absolute error from the clipped real
reference is bounded by $192h + \tfrac12 2^{-F} = \tfrac{11}{16}2^{-F}$.
An independently rounded reference lies within another half LSB. Since
both output codes are integers, their difference is at most one.
