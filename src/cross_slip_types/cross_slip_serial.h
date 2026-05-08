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
