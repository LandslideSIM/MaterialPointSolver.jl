//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::MohrCoulomb<Tdim>::compute_stress(
    const Vector6d &stress, const Vector6d &dstrain,
    const ParticleBase<Tdim> *ptr, mpm::dense_map *state_vars) {
    // Get equivalent plastic deviatoric strain
    const double pdstrain = (*state_vars).at("pdstrain");
    // Update MC parameters using a linear softening rule
    if (softening_ && pdstrain > pdstrain_peak_) {
        if (pdstrain < pdstrain_residual_) {
            (*state_vars).at("phi") =
                phi_residual_ +
                ((phi_peak_ - phi_residual_) * (pdstrain - pdstrain_residual_) /
                 (pdstrain_peak_ - pdstrain_residual_));
            (*state_vars).at("psi") =
                psi_residual_ +
                ((psi_peak_ - psi_residual_) * (pdstrain - pdstrain_residual_) /
                 (pdstrain_peak_ - pdstrain_residual_));
            (*state_vars).at("cohesion") =
                cohesion_residual_ + ((cohesion_peak_ - cohesion_residual_) *
                                      (pdstrain - pdstrain_residual_) /
                                      (pdstrain_peak_ - pdstrain_residual_));
        } else {
            (*state_vars).at("phi") = phi_residual_;
            (*state_vars).at("psi") = psi_residual_;
            (*state_vars).at("cohesion") = cohesion_residual_;
        }
    }
    //-------------------------------------------------------------------------
    // Elastic-predictor stage: compute the trial stress
    Vector6d trial_stress = stress + (this->de_ * dstrain);
    // Compute stress invariants based on trial stress
    this->compute_stress_invariants(trial_stress, state_vars);
    // Compute yield function based on the trial stress
    Eigen::Matrix<double, 2, 1> yield_function_trial;
    auto yield_type_trial =
        this->compute_yield_state(&yield_function_trial, (*state_vars));
    // Return the updated stress in elastic state
    if (yield_type_trial == mpm::mohrcoulomb::FailureState::Elastic)
        return trial_stress;
    //-------------------------------------------------------------------------
    // Plastic-corrector stage: correct the stress back to the yield surface
    // Define tolerance of yield function
    const double Tolerance = 1E-1;
    // Compute plastic multiplier based on trial stress (Lambda trial)
    double softening_trial = 0.;
    double dp_dq_trial = 0.;
    Vector6d df_dsigma_trial = Vector6d::Zero();
    Vector6d dp_dsigma_trial = Vector6d::Zero();
    this->compute_df_dp(yield_type_trial, state_vars, trial_stress,
                        &df_dsigma_trial, &dp_dsigma_trial, &dp_dq_trial,
                        &softening_trial);
    double yield_trial = 0.;
    if (yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile)
        yield_trial = yield_function_trial(0);
    if (yield_type_trial == mpm::mohrcoulomb::FailureState::Shear)
        yield_trial = yield_function_trial(1);
    double lambda_trial =
        yield_trial /
        ((df_dsigma_trial.transpose() * de_).dot(dp_dsigma_trial.transpose()) +
         softening_trial);
    // Compute stress invariants based on stress input
    this->compute_stress_invariants(stress, state_vars);
    // Compute yield function based on stress input
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));
    // Initialise value of yield function based on stress
    double yield{std::numeric_limits<double>::max()};
    if (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
        yield = yield_function(0);
    if (yield_type == mpm::mohrcoulomb::FailureState::Shear)
        yield = yield_function(1);
    // Compute plastic multiplier based on stress input (Lambda)
    double softening = 0.;
    double dp_dq = 0.;
    Vector6d df_dsigma = Vector6d::Zero();
    Vector6d dp_dsigma = Vector6d::Zero();
    this->compute_df_dp(yield_type, state_vars, stress, &df_dsigma, &dp_dsigma,
                        &dp_dq, &softening);
    const double lambda =
        ((df_dsigma.transpose() * this->de_).dot(dstrain)) /
        (((df_dsigma.transpose() * this->de_).dot(dp_dsigma)) + softening);
    // Initialise updated stress
    Vector6d updated_stress = trial_stress;
    // Initialise incremental of plastic deviatoric strain
    double dpdstrain = 0.;
    // Correction stress based on stress
    if (fabs(yield) < Tolerance) {
        // Compute updated stress
        updated_stress -= (lambda * this->de_ * dp_dsigma);
        // Compute incremental of plastic deviatoric strain
        dpdstrain = lambda * dp_dq;
    } else {
        // Compute updated stress
        updated_stress -= (lambda_trial * this->de_ * dp_dsigma_trial);
        // Compute incremental of plastic deviatoric strain
        dpdstrain = lambda_trial * dp_dq_trial;
    }

    // Define the maximum iteration step
    const int itr_max = 100;
    // Correct the stress again
    for (unsigned itr = 0; itr < itr_max; ++itr) {
        // Check the update stress
        // Compute stress invariants based on updated stress
        this->compute_stress_invariants(updated_stress, state_vars);
        // Compute yield function based on updated stress
        yield_type_trial =
            this->compute_yield_state(&yield_function_trial, (*state_vars));
        // Check yield function
        if (yield_function_trial(0) < Tolerance &&
            yield_function_trial(1) < Tolerance) {
            break;
        }
        // Compute plastic multiplier based on updated stress
        this->compute_df_dp(yield_type_trial, state_vars, updated_stress,
                            &df_dsigma_trial, &dp_dsigma_trial, &dp_dq_trial,
                            &softening_trial);
        if (yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile)
            yield_trial = yield_function_trial(0);
        if (yield_type_trial == mpm::mohrcoulomb::FailureState::Shear)
            yield_trial = yield_function_trial(1);
        // Compute plastic multiplier based on updated stress
        lambda_trial = yield_trial / ((df_dsigma_trial.transpose() * de_)
                                          .dot(dp_dsigma_trial.transpose()) +
                                      softening_trial);
        // Correct stress back to the yield surface
        updated_stress -= (lambda_trial * this->de_ * dp_dsigma_trial);
        // Update incremental of plastic deviatoric strain
        dpdstrain += lambda_trial * dp_dq_trial;
    }
    // Compute stress invariants based on updated stress
    this->compute_stress_invariants(updated_stress, state_vars);
    // Update plastic deviatoric strain
    (*state_vars).at("pdstrain") += dpdstrain;

    return updated_stress;
}