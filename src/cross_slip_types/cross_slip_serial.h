/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This module implements cross-slip for FCC crystals in serial fashion.
 *  It is a direct translation in ExaDiS of ParaDiS source file
 *  ParaDiS/src/CrossSlipFCC.cc.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_SERIAL_H
#define EXADIS_CROSS_SLIP_SERIAL_H

#include "force.h"
#include "cross_slip.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       BCCCrossSlipParams
 *                  Parameters for thermally-activated BCC cross-slip.
 *                  Temperature model: T = kT * pstrain + bT (adiabatic heating).
 *
 *-------------------------------------------------------------------------*/
struct BCCCrossSlipParams {
    double kT          = 0.0;  // heating slope [K/strain]
    double bT          = 300.0;// initial temperature [K]
    double delta_H_cs  = 0.0;  // activation enthalpy at zero stress [eV]
    double tau_P_cs    = 0.0;  // Peierls stress on cross-slip plane [Pa]
    double p_shape     = 0.5;  // Kocks-Mecking shape parameter p
    double q_shape     = 1.5;  // Kocks-Mecking shape parameter q
    double delta_S_cs  = 0.0;  // activation entropy [eV/K]
    double omega_D     = 0.0;  // Debye attempt frequency [s^-1]
    double eps_dot_sim = 0.0;  // simulated strain rate [s^-1]
    double eps_dot_exp = 0.0;  // experimental strain rate [s^-1]
    double L0_ref      = 0.0;  // reference dislocation length [m]
    double tau_f_cs    = 0.0;  // cross-slip plane friction stress [Pa]
};

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlipSerial
 *
 *-------------------------------------------------------------------------*/
class CrossSlipSerial : public CrossSlip {
private:
    Force* force;

public:
    BCCCrossSlipParams bcc_params;

    CrossSlipSerial(System* system, Force* _force) : force(_force) {}
    CrossSlipSerial(System* system, Force* _force, BCCCrossSlipParams _bcc_params)
        : force(_force), bcc_params(_bcc_params) {}

    void handle(System* system);

    const char* name() { return "CrossSlipSerial"; }
};

} // namespace ExaDiS

#endif
