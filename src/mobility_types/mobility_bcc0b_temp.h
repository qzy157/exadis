/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *	Junjie
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_BCC0B_TEMP_H
#define EXADIS_MOBILITY_BCC0B_TEMP_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityBCC0b_temp
 *                  Temperature-dependent BCC mobility where Bscrew and Bedge
 *                  are functions of plastic strain (used as a proxy for
 *                  adiabatic heating: T = kT * pstrain + bT).
 *
 *-------------------------------------------------------------------------*/
struct MobilityBCC0b_temp
{
    bool non_linear = false;
    double Beclimbj;
    double Beclimb2;
    double Beclimb;
    double vmax, vscale;
    double Fedge, Fscrew;
    double kT, bT;

    struct Params {
        double Mclimb = -1.0;
        double kT = 0.0, bT = 0.0;
        double Fedge = 0.0, Fscrew = 0.0;
        double vmax = -1.0;
        Params() {}
        Params(double _Mclimb) {
            Mclimb = _Mclimb;
        }
        Params(double _Mclimb, double _vmax) {
            Mclimb = _Mclimb;
            vmax = _vmax;
        }
        Params(double _Mclimb, double _kT, double _bT,
               double _Fedge, double _Fscrew, double _vmax) {
            Mclimb = _Mclimb;
            kT = _kT;
            bT = _bT;
            Fedge = _Fedge;
            Fscrew = _Fscrew;
            vmax = _vmax;
        }
        Params(Dict paramslist) {
            for (auto const& [key, val] : paramslist) {
                std::string name = dict::get_key(key);
                if      (name == "Mclimb") Mclimb = dict::get_val<double>(val);
                else if (name == "kT")     kT     = dict::get_val<double>(val);
                else if (name == "bT")     bT     = dict::get_val<double>(val);
                else if (name == "Fedge")  Fedge  = dict::get_val<double>(val);
                else if (name == "Fscrew") Fscrew = dict::get_val<double>(val);
                else if (name == "vmax")   vmax   = dict::get_val<double>(val);
                else ExaDiS_fatal("Error: unknown MobilityBCC0b_temp input parameter '%s'\n", name.c_str());
            }
        }
    };

    MobilityBCC0b_temp(System* system, Params& params)
    {
        if (system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityBCC0b_temp() must be used with BCC crystal type\n");

        if (params.Mclimb < 0.0 || params.Fedge < 0 || params.Fscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityBCC0b_temp() parameter values\n");

        Beclimb = 1.0 / params.Mclimb;
        vmax    = params.vmax;
        vscale  = system->params.burgmag;
        Fedge   = params.Fedge;
        Fscrew  = params.Fscrew;
        kT      = params.kT;
        bT      = params.bT;

        if (Fedge > 1e-5 || Fscrew > 1e-5)
            non_linear = true;

        Beclimbj = Beclimb;
        Beclimb2 = Beclimb * Beclimb;
    }

    KOKKOS_INLINE_FUNCTION
    double compute_Bscrew(double pstrain, double tau_rss)
    {
        double A    = 1.8804e-5;
        double dH0  = 0.257;       // eV, activation enthalpy at 0 K
        double p    = 0.5614;
        double q    = 0.5987;
        double tau_p = 814.91e6;   // Peierls stress in Pa

        double Tm = 1589.0 + 273.15;
        double T  = kT * pstrain + bT;
        if (T <= 0.0) T = 1.0; // guard: bT must be a positive initial temperature (K)
        double kB = 8.617333262145e-5; // eV/K

        if (tau_rss >= tau_p) {
            return A * T;
        } else {
            double dG = dH0 * (pow((1.0 - pow(tau_rss / tau_p, p)), q) - T / Tm);
            if (dG < 0.0) dG = 0.0;
            return A * T * exp(dG / (kB * T));
        }
    }

    KOKKOS_INLINE_FUNCTION
    double compute_Bedge(double pstrain)
    {
        double A = 1e-7;
        double T = kT * pstrain + bT;
        if (T <= 0.0) T = 1.0;
        return A * T;
    }

    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_velocity(System* system, N* net, const int& i, const Vec3& fi)
    {
        double pstrain  = system->pstrain;
        double Bedge    = compute_Bedge(pstrain);
        double invBedge2 = 1.0 / (Bedge * Bedge);

        auto nodes = net->get_nodes();
        auto segs  = net->get_segs();
        auto conn  = net->get_conn();
        auto cell  = net->cell;

        Vec3 vi(0.0);

        if (conn[i].num >= 2 && nodes[i].constraint != PINNED_NODE) {

            int linejunc = 0;
            Vec3 tjunc(0.0);

            if (system->params.split3node) {
                int binaryjunc = 0;
                int planarjunc = 0;
                binaryjunc = BCC_binary_junction_node(system, net, i, tjunc, &planarjunc);
                binaryjunc = (binaryjunc > -1);

                if (binaryjunc) {
                    if (planarjunc) {
                        linejunc = 1;
                    } else {
                        int unzipping = (dot(fi, tjunc) > 0.0);
                        if (unzipping) linejunc = 1;
                    }
                }
            }

            double eps = 1e-12;
            Mat33 Btotal = Mat33().zero();
            double FricForce = 0.0;

            Vec3 r1 = nodes[i].pos;
            int numNonZeroLenSegs = 0;
            for (int j = 0; j < conn[i].num; j++) {

                int k     = conn[i].node[j];
                int s     = conn[i].seg[j];
                int order = conn[i].order[j];

                Vec3 burg   = order * segs[s].burg;
                double bMag  = burg.norm();
                double bMag2 = bMag * bMag;
                double invbMag2 = 1.0 / bMag2;

                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2 - r1;
                double mag = dr.norm();
                if (mag < eps) continue;
                numNonZeroLenSegs++;

                double halfMag = 0.5 * mag;
                double invMag  = 1.0 / mag;
                dr = invMag * dr;

                double costheta  = dot(dr, burg);
                double costheta2 = (costheta * costheta) * invbMag2;

                double dangle = 1.0 / bMag * fabs(costheta);

                double Bscrew    = compute_Bscrew(pstrain, fi.norm() / bMag / mag);
                double Bscrew2   = Bscrew * Bscrew;
                double Bline     = 1.0e-2 * MIN(Bscrew, Bedge);
                double BlmBsc    = Bline - Bscrew;
                double BlmBecl   = Bline - Beclimbj;
                double invBscrew2 = 1.0 / Bscrew2;

                if (bMag > 1.0 + eps) {
                    // [0 0 1] junction arms
                    if (linejunc == 1) {
                        Btotal[0][0] += halfMag * Bline;
                        Btotal[1][1] += halfMag * Bline;
                        Btotal[2][2] += halfMag * Bline;
                        continue;
                    }
                    if (conn[i].num == 2) {
                        Btotal[0][0] += halfMag * Beclimbj;
                        Btotal[1][1] += halfMag * Beclimbj;
                        Btotal[2][2] += halfMag * Beclimbj;
                    } else {
                        Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBecl + Beclimbj);
                        Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBecl);
                        Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBecl);
                        Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBecl + Beclimbj);
                        Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBecl);
                        Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBecl + Beclimbj);
                    }
                } else {
                    // Non-[0 0 1] arm: friction and screw drag
                    double fricStress = Fedge + (Fscrew - Fedge) * dangle;
                    FricForce += fricStress * bMag * mag;
                    Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBsc + Bscrew);
                    Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBsc);
                    Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBsc);
                    Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBsc + Bscrew);
                    Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBsc);
                    Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBsc + Bscrew);

                    // Correction for non-screw character
                    if ((1.0 - costheta2) > eps) {
                        double invsqrt1mcostheta2 = 1.0 / sqrt((1.0 - costheta2) * bMag2);
                        Vec3 nr = cross(burg, dr);
                        nr = invsqrt1mcostheta2 * nr;
                        Vec3 mr = cross(nr, dr);

                        if (linejunc == 1) {
                            nr -= dot(nr, tjunc) * tjunc;
                            nr = nr.normalized();
                        }

                        double Bglide  = sqrt(invBedge2 + (invBscrew2 - invBedge2) * costheta2);
                        Bglide = 1.0 / Bglide;
                        double Bclimb  = sqrt(Beclimb2 + (Bscrew2 - Beclimb2) * costheta2);
                        double BclmBsc = Bclimb - Bscrew;
                        double BglmBsc = Bglide - Bscrew;

                        Btotal[0][0] += halfMag * (nr.x*nr.x * BclmBsc + mr.x*mr.x * BglmBsc);
                        Btotal[0][1] += halfMag * (nr.x*nr.y * BclmBsc + mr.x*mr.y * BglmBsc);
                        Btotal[0][2] += halfMag * (nr.x*nr.z * BclmBsc + mr.x*mr.z * BglmBsc);
                        Btotal[1][1] += halfMag * (nr.y*nr.y * BclmBsc + mr.y*mr.y * BglmBsc);
                        Btotal[1][2] += halfMag * (nr.y*nr.z * BclmBsc + mr.y*mr.z * BglmBsc);
                        Btotal[2][2] += halfMag * (nr.z*nr.z * BclmBsc + mr.z*mr.z * BglmBsc);
                    }
                }
            }  // end loop over arms

            Btotal[1][0] = Btotal[0][1];
            Btotal[2][0] = Btotal[0][2];
            Btotal[2][1] = Btotal[1][2];
            FricForce /= 2.0;

            if (numNonZeroLenSegs > 0) {
                Mat33 invDragMatrix = Btotal.inverse();

                Vec3 f = fi;
                if (FricForce > eps) {
                    double fmag = f.norm();
                    if (fmag > FricForce) {
                        f -= FricForce / fmag * f;
                    } else {
                        f = Vec3(0.0);
                    }
                }

                vi = invDragMatrix * f;
                if (vmax > 0.0)
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }

        return vi;
    }

    static constexpr const char* name = "MobilityBCC0b_temp";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityBCC0b_temp> BCC_0B_TEMP;
}

} // namespace ExaDiS

EXADIS_MOBILITY(MobilityBCC0b_temp, BCC_0B_TEMP)

#endif
