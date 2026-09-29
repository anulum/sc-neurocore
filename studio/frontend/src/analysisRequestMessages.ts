// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — what a refused analysis request tells the reader, in words

/**
 * Sentences for the identifiers an analysis request is refused with.
 *
 * The selection and request builders refuse with stable identifiers such as
 * `analysis_selection_heatmap_param_x_blank`, which tests and logs rely on.
 * The header showed that identifier to the reader as the explanation. The
 * identifier stays the contract; this module says what it means and what to
 * do about it.
 */

const MESSAGES: Readonly<Record<string, string>> = {
  analysis_selection_sweep_param_blank: "Choose a parameter in Sweep X to run a bifurcation sweep.",
  analysis_selection_sweep_param_missing:
    "The Sweep X parameter is not one of this model's parameters; choose it again.",
  analysis_selection_sweep_value_invalid:
    "The Sweep X parameter has no finite value; set it before sweeping.",
  analysis_selection_heatmap_param_x_blank: "Choose parameters in Sweep X and Sweep Y to run a 2-D sweep.",
  analysis_selection_heatmap_param_x_missing:
    "The Sweep X parameter is not one of this model's parameters; choose it again.",
  analysis_selection_heatmap_value_x_invalid:
    "The Sweep X parameter has no finite value; set it before sweeping.",
  analysis_selection_heatmap_param_y_blank: "Choose a second parameter in Sweep Y to run a 2-D sweep.",
  analysis_selection_heatmap_param_y_missing:
    "The Sweep Y parameter is not one of this model's parameters; choose it again.",
  analysis_selection_heatmap_value_y_invalid:
    "The Sweep Y parameter has no finite value; set it before sweeping.",
  analysis_selection_heatmap_axes_identical: "Sweep X and Sweep Y must be different parameters.",
  analysis_request_sweep_param_blank: "Choose a parameter in Sweep X to run a bifurcation sweep.",
  analysis_request_sweep_value_invalid: "The swept parameter has no finite value.",
  analysis_request_heatmap_param_blank: "Choose parameters in Sweep X and Sweep Y to run a 2-D sweep.",
  analysis_request_heatmap_value_invalid: "A swept parameter has no finite value.",
  analysis_request_heatmap_axes_identical: "Sweep X and Sweep Y must be different parameters.",
  analysis_request_current_invalid: "The input current must be a finite number.",
  analysis_request_dt_invalid: "The time step must be a positive, finite number.",
  analysis_request_duration_invalid: "The duration must be a positive, finite number.",
  analysis_request_model_params_invalid: "Every model parameter must be a finite number.",
  analysis_request_ode_params_invalid: "Every equation parameter must be a finite number.",
  analysis_request_ode_init_invalid: "Every initial value must be a finite number.",
};

/**
 * Say in words why an analysis request cannot be sent.
 *
 * @param code - The identifier the request was refused with.
 * @returns A sentence for the reader; an identifier this module does not know
 *   is named rather than hidden, so nothing is ever explained away.
 */
export function analysisRequestMessage(code: string): string {
  return MESSAGES[code] ?? `The analysis request was refused (${code}).`;
}

const JOB_FAILURES: Readonly<Record<string, string>> = {
  ModelSimulationFailure:
    "The model's simulation failed at one of the analysed points: a state left the model's " +
    "safety bounds or became non-finite. Try a narrower range, a smaller time step or a " +
    "different drive.",
  ModelInputError: "The server refused a model parameter or input for this analysis.",
  cancelled: "The analysis was cancelled.",
};

/**
 * Say in words why an analysis job failed.
 *
 * The job worker reports only an error class or a short status (it never
 * returns a message that could carry a path), and the header showed that
 * class name, such as `ModelSimulationFailure`, as the explanation.
 *
 * @param code - The class name or status the job failed with.
 * @returns A sentence; an unknown code is named rather than hidden.
 */
export function analysisJobErrorMessage(code: string): string {
  return JOB_FAILURES[code] ?? `The analysis failed on the server (${code}).`;
}
