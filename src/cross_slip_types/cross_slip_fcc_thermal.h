/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Thermally-activated cross-slip for FCC crystals.
 *  Reference: Hussein, Ahmed M., et al. Acta Materialia 85 (2015): 180-190.
 *
 *  The algorithm groups connected screw segments into "chains" and evaluates
 *  thermodynamic transition probabilities driven by Escaig stress differences
 *  between the glide and cross-slip planes. Two chain types are supported:
 *    - Bulk:         chains fully inside the simulation volume
 *    - Intersection: chains terminating at dislocation junctions (Hirth,
 *                    Glide lock, LC lock)
 *
 *  Thermal probability:
 *    P = nu * dt * (L / L_ref) * exp(-(E_a - V_a * dSigma_Escaig) / (kB * T))
 *
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_FCC_THERMAL_H
#define EXADIS_CROSS_SLIP_FCC_THERMAL_H//头文件保护双保险，防止重复包含

#include <random>//引入 C++11 标准库的随机数生成头文件
#include <cmath>//引入 C++11 标准库的数学函数头文件
#include <vector>//引入 C++11 标准库的向量容器头文件
#include "force.h"//引入项目中的力相关头文件
#include "cross_slip.h"//

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:    CrossSlipFCCThermal
 *              Thermally-activated FCC cross-slip (serial implementation).
 *
 *-------------------------------------------------------------------------*/
class CrossSlipFCCThermal : public CrossSlip {
public://继承自抽象基类 CrossSlip，意味着它必须实现基类的虚函数 handle() 和 name()

    /*-----------------------------------------------------------------------
     *    Struct:   Params
     *              Material and cross-slip parameters. All activation
     *              energies are in Joules, stacking fault energies in J/m^2,
     *              lengths in units of burgmag.
     *---------------------------------------------------------------------*/
    struct Params {
        double temperature   = 300.0; ///< Temperature [K]
        int    evalFrequency = 1;     ///< Steps between evaluations,每隔几步评估依次交滑移

        // --- Bulk cross-slip ---
        double bulkActivationEnergy       = 0.8*1.602E-19;  ///< E_a [J]
        double bulkActivationVolume       = 20.0;           ///< V_a factor (need to multiple b^3)
        double bulkAttemptFrequency       = 5.0E17;         ///< nu [1/s] 尝试频率
        double bulkReferenceLength        = 1000.0;         ///< L_ref [b] 参考长度，单位b

        // --- Hirth lock (intersection) ---
        double hirthActivationEnergy       = 0.2*1.602E-19;
        double hirthActivationVolume       = 20.0;
        double hirthAttemptFrequency       = 5.0E17;
        double hirthReferenceLength        = 1000.0;
        double hirthEffectiveLength        = 2.5e-9;        ///< 2.5nm (need to divide burgmag when use) （固定有效长度 2.5 nm，注释说用时除以 burgmag 转成 b 单位）

        // --- Glide lock (intersection) ---
        double glideLockActivationEnergy       = 0.5*1.602E-19;
        double glideLockActivationVolume       = 20.0;
        double glideLockAttemptFrequency       = 5.0E17;
        double glideLockReferenceLength        = 1000.0;
        double glideLockEffectiveLength        = 2.5e-9;    ///< 2.5nm (need to divide burgmag when use)

        // --- LC lock (intersection) ---
        double lcLockActivationEnergy       = 0.6*1.602E-19;
        double lcLockActivationVolume       = 20.0;
        double lcLockAttemptFrequency       = 5.0E17;
        double lcLockReferenceLength        = 1000.0;
        double lcLockEffectiveLength        = 2.5e-9;       ///< 2.5nm (need to divide burgmag when use)

        /// Segment is considered screw if its angle with b is within this threshold [deg]
        double screwAngleTolerance = 15.0;//螺旋角容差，单位为度，表示如果一个段与 Burgers 向量的夹角在这个范围内，就被认为是螺旋段

        Params() = default;
    };

private:
    Force*  force;//存了 Force 指针（注意：后面其实没用到）
    Params  params;
    int     eval_counter = 0;//步数计数器

    std::mt19937 rng;//梅森旋转随机数引擎
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};// [0,1) 均匀分布

    enum MechanismType { Bulk, Intersection };//交滑移机制类型：Bulk（完全在体积内的链）和 Intersection（在位错交点处的链）
    enum LockType  { Hirth, GlideLock, LCLock, UnknownLock };//交点锁类型：Hirth 锁、Glide 锁、LC 锁和未知锁

    /*-----------------------------------------------------------------------
     *    Struct:   ScrewChain
     *              Connected screw-character segments sharing the same
     *              Burgers vector and {111} glide plane.
     *---------------------------------------------------------------------*/
    struct ScrewChain {
        std::vector<int> node_ids;//链上节点 id（有 n+1 个）
        std::vector<int> seg_ids;//链上 segment id（有 n 个）
        Vec3 burg;//实验室系 Burgers 矢量
        Vec3 glide_plane;//当前滑移面法向
        MechanismType mechanism = Bulk;//交滑移机制类型默认为 Bulk
        LockType  junction  = UnknownLock;//交点锁类型，默认为未知锁
        Vec3 junction_burg;//结另一臂的合 Burgers 矢量
        Vec3 junction_plane;//结另一臂的面
    };

    // -----------------------------------------------------------------------
    //  Classify a vector by FCC crystallographic family
    // -----------------------------------------------------------------------
    static bool is_100_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        int nonzero = (fabs(u.x) > tol) + (fabs(u.y) > tol) + (fabs(u.z) > tol);
        return nonzero == 1;
    }//判断一个向量是否属于 FCC 晶体的 {100} 家族，即是否只有一个分量显著非零 ⟨100⟩ 判据：归一化后只有一个分量非零（比如 [1,0,0]）。(条件)+(条件) 利用 bool→int 隐式转换数非零分量个数。

    static bool is_110_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        if (ax < tol && fabs(ay - az) < tol && ay > tol) return true;
        if (ay < tol && fabs(ax - az) < tol && ax > tol) return true;
        if (az < tol && fabs(ax - ay) < tol && ax > tol) return true;
        return false;
    }//⟨110⟩ 判据：一个分量为零，另两个绝对值相等且非零（如 [1,1,0]）。三个 if 覆盖零分量落在 x/y/z 的三种情况。

    static bool is_111_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        return (fabs(ax - ay) < tol && fabs(ay - az) < tol && ax > tol);
    }//⟨111⟩ 判据：三个分量绝对值相等且非零（如 [1,1,1]）。直接比较三个分量两两之间的差是否小于容差。

    static bool is_112_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        double vals[3] = {ax, ay, az};
        double vmax = (ax > ay ? ax : ay);
        vmax = (vmax > az ? vmax : az);
        if (vmax < tol) return false;
        int n_large = 0, n_small = 0;
        for (int i = 0; i < 3; i++) {
            if      (fabs(vals[i] - vmax) < tol)          n_large++;//等于最大值
            else if (fabs(2.0 * vals[i] - vmax) < tol)    n_small++;//等于最大值的一半
            else if (vals[i] < tol) {}//约等于0也算（容错）
            else return false;//不满足上述条件则不属于 ⟨112⟩ 家族
        }
        return (n_large == 1 && n_small == 2);
    }//⟨112⟩ 判据：一个分量最大（如 2），另两个分量约等于最大值的一半（如 1），且都非零（如 [2,1,1]）。先找出最大值，再统计有多少分量接近最大值，多少分量接近最大值的一半，最后检查是否满足 1 个接近最大值，2 个接近一半的条件。

    // -----------------------------------------------------------------------
    //  Returns the cross-slip plane normal for an FCC 1/2<110> dislocation.
    //  Given glide plane n and Burgers vector b, returns the other {111} plane
    //  containing b by flipping the sign of the zero component of b in n.
    // -----------------------------------------------------------------------
    static Vec3 get_crossslip_plane(const Vec3& glide_plane_normal,
                                    const Vec3& burg) {
        const double tol = 1e-6;
        Vec3 bn = burg.normalized();
        if (fabs(dot(bn, glide_plane_normal)) > tol) return Vec3(0.0);//如果 Burgers 向量与滑移面法向的点积不接近零，说明它们不垂直，返回零向量表示无法确定交滑移面
        if (fabs(bn.x) < tol) return Vec3(-glide_plane_normal.x,  glide_plane_normal.y,  glide_plane_normal.z);
        if (fabs(bn.y) < tol) return Vec3( glide_plane_normal.x, -glide_plane_normal.y,  glide_plane_normal.z);
        if (fabs(bn.z) < tol) return Vec3( glide_plane_normal.x,  glide_plane_normal.y, -glide_plane_normal.z);
        return Vec3(0.0);//如果 Burgers 向量的三个分量都不接近零，说明它不符合 1/2<110> 的特征，无法确定交滑移面，返回零向量。
    }//对 1/2⟨110⟩ 位错，含 b 的 {111} 面有两张，当前面是 glide，另一张就是 cross-slip 面。算法：b 哪个分量为零，就把法向那个分量翻号。例如 b=[1,-1,0]（z 分量为零），glide=(1,1,1)，翻 z 得 (1,1,-1)，正好是另一张含 b 的 {111}。注意我上轮提的隐患就在这里——它要求 b 是轴对齐的，即只在晶体坐标系成立，而传进来的是实验室系向量。

    // -----------------------------------------------------------------------
    //  Escaig stress component from accumulated chain force.
    // -----------------------------------------------------------------------
    static double compute_escaig_stress(const Vec3& force,
                                        const Vec3& burg_dir,
                                        const Vec3& plane_normal,
                                        double      burg_mag,
                                        double      total_length) {
        if (burg_mag < 1e-15 || total_length < 1e-15) return 0.0;
        Vec3 escaig_dir = cross(plane_normal, burg_dir);//Escaig 方向垂直于 Burgers 向量和面法向的叉积
        double len = escaig_dir.norm();
        if (len < 1e-10) return 0.0;
        return dot(force, escaig_dir / len) / (burg_mag * total_length);
    }//Escaig 应力的计算：首先计算 Escaig 方向（垂直于 Burgers 向量和面法向的叉积），然后将链上累积的力投影到 Escaig 方向上，并除以 Burgers 模长和链长，得到平均 Escaig 应力。注意这里也有隐患：如果 Burgers 向量和面法向几乎平行，Escaig 方向的模长会很小，可能导致数值不稳定。 Escaig 应力作用在刃分量上、驱动分位错间距变化。cross(n,b) 给出面内垂直于 b 的方向（刃方向），把合力投影到该单位方向，再除以 b·L 归一成应力。这里除了 burg_mag，而后面 Schmid 没除——就是我上轮说的量纲不一致点。

    // -----------------------------------------------------------------------
    //  Arrhenius cross-slip probability per time step (Hussein et al. Eq. 2):
    //  P = nu * dt * (L / L_ref) * exp(-(E_a - V_a * dSigma_E) / (kB * T))
    // -----------------------------------------------------------------------
    static double compute_thermal_probability(double E_a, double V_a,
                                              double dSigma_E, double temperature,
                                              double attempt_frequency, double dt,
                                              double chain_length, double reference_length) {
        const double kB = 1.3806503e-23;
        double exponent = -(E_a - V_a * dSigma_E) / (kB * temperature);
        return attempt_frequency * dt * (chain_length / reference_length) * exp(exponent);
    }//根据 Arrhenius 公式计算热激活的交滑移概率。输入参数包括激活能 E_a、激活体积 V_a、Esclaig 应力差 dSigma_E、温度、尝试频率、时间步长、链长和参考长度。输出是一个概率值，表示在给定条件下发生交滑移的概率。 就是注释那条 Arrhenius 公式。V_a*dSigma_E 是应力对势垒的降低量，dSigma_E>0 时指数变大、概率升高。前面乘 ν·dt·(L/L_ref)：长链有更多成核位点、概率正比于长度。

    // -----------------------------------------------------------------------
    //  Builds all screw chain candidates from physical dislocation links.
    //  Applies FCC crystallographic filters and the screw angle criterion.
    // -----------------------------------------------------------------------
    std::vector<ScrewChain> build_screw_chains(System* system,
                                               SerialDisNet* network)
    {
        const double tol  = 1e-6;
        const double scos = cos(params.screwAngleTolerance * M_PI / 180.0);//螺旋角容差的余弦值，用于判断段是否具有足够的螺旋特征。比如 15 度对应 cos(15°) ≈ 0.9659，意味着段与 Burgers 向量的夹角必须小于 15 度才能被认为是螺旋段。

        SerialDisNet::DisLinks dislinks = network->physical_links();//获取物理位错链信息，包括链上节点和段的 id 列表。每条链至少包含一个段和两个节点。 取所有"物理链"（两结点之间的一串segment）
        std::vector<ScrewChain> chains;//存储满足条件的螺旋链

        for (int l = 0; l < dislinks.number_of_links; l++) {
            const auto& snodes = dislinks.links_nodes[l];//链上节点 id 列表
            const auto& ssegs  = dislinks.links_segs[l];//链上段 id 列表
            if (ssegs.empty()) continue;
            int nseg = (int)ssegs.size();//链上段的数量 遍历每条物理链。对于每条链，首先检查是否为空链（没有段），如果是则跳过。然后获取链上节点和段的 id 列表，并计算段的数量 nseg。

            // Burgers vector, sign-corrected for traversal direction
            int s0 = ssegs[0];//链上第一个段的 id，根据这个段的 Burgers 向量来确定整个链的 Burgers 向量。由于链上段的方向可能不一致，需要根据第一个段的节点顺序来修正 Burgers 向量的符号。具体来说，取第一个段 s0 的两个节点 id，判断哪个节点在前（snodes[0]）哪个在后（snodes[1]），如果 s0 的 n2 是 snodes[0]，说明第一个段的方向与节点列表的顺序相反，需要将 Burgers 向量取反。
            int n0 = snodes[0];//链上第一个节点的 id
            Vec3 burg = network->segs[s0].burg;//获取第一个段的 Burgers 向量
            if (network->segs[s0].n2 == n0) burg = -burg;//如果第一个段的第二个节点是链上第一个节点，说明段的方向与节点列表的顺序相反，需要将 Burgers 向量取反。这样就确保了整个链的 Burgers 向量与节点列表的顺序一致。 取第一个 segment 的 Burgers 矢量。位错 segment 有方向（n1→n2），若遍历起点 n0 恰好是 segment 的终点 n2，说明方向相反，b 取负，保证沿链方向一致。

            // Reject links with inconsistent glide planes
            Vec3 plane0 = network->segs[s0].plane;//获取第一个段的滑移面法向量，作为整个链的参考滑移面。后续会检查链上其他段的滑移面是否与这个参考面一致。
            if (plane0.norm() < tol) continue;//如果第一个段的滑移面法向量的模长小于容差，说明这个段没有有效的滑移面信息，无法用来判断链的滑移面一致性，因此跳过这个链。否则，将这个参考滑移面归一化，后续会用它来检查链上其他段的滑移面是否一致。
            plane0 = plane0.normalized();//将第一个段的滑移面法向量归一化，得到单位向量 plane0 作为整个链的参考滑移面。后续会检查链上其他段的滑移面是否与这个参考面一致。对于 FCC 位错，所有段应该共享同一 {111} 面，因此它们的法向量应该相同或相反（即 plane0 或 -plane0）。

            bool consistent_plane = true;//标志位，表示链上所有段的滑移面是否与参考面一致。初始值为 true，后续会检查链上每个段的滑移面，如果发现不一致的情况（既不接近 plane0 也不接近 -plane0），就将这个标志置为 false，并跳出循环。
            for (int k = 0; k < nseg; k++) {
                int sk = ssegs[k];
                Vec3 p = network->segs[sk].plane;
                if (p.norm() < tol) { consistent_plane = false; break; }
                p = p.normalized();
                if ((p - plane0).norm() > tol && (p + plane0).norm() > tol) {
                    consistent_plane = false; break;
                }//对于链上每个段，获取它的滑移面法向量 p，并检查它是否与参考面 plane0 一致。具体来说，首先检查 p 的模长是否有效，如果小于容差，说明这个段没有有效的滑移面信息，直接标记为不一致并跳出循环。否则，将 p 归一化后，检查它与 plane0 的差和与 -plane0 的差是否都小于容差。如果两者都不接近，说明这个段的滑移面与参考面不一致，也标记为不一致并跳出循环。只有当链上所有段的滑移面都接近 plane0 或 -plane0 时，consistent_plane 才保持为 true，表示这个链满足 FCC 位错的滑移面一致性要求。
            }
            if (!consistent_plane) continue;//如果链上存在滑移面不一致的段，说明这个链不满足 FCC 位错的特征，直接跳过这个链，不将它纳入后续的螺旋链构建和分析中。

            // FCC filter: Burgers must be 1/2<110>, plane must be {111}
            Vec3 burg_crystal  = system->crystal.Rinv * burg.normalized();
            if (!is_110_family(burg_crystal)) continue;//将 Burgers 向量从实验室坐标系转换到晶体坐标系，得到 burg_crystal。然后检查它是否属于 FCC 晶体的 ⟨110⟩ 家族，即是否满足 1/2<110> 的特征。如果不满足，说明这个链的 Burgers 向量不符合 FCC 位错的特征，直接跳过这个链，不将它纳入后续的螺旋链构建和分析中。
            Vec3 plane_crystal = system->crystal.Rinv * plane0;//FCC 过滤。关键：先用 Rinv 把向量转到晶体系再判族，这是正确做法——但对照前面 get_crossslip_plane 用的是实验室系，这就是不一致的根源。
            if (!is_111_family(plane_crystal)) continue;//将参考滑移面法向 plane0 从实验室坐标系转换到晶体坐标系，得到 plane_crystal。然后检查它是否属于 FCC 晶体的 ⟨111⟩ 家族，即是否满足 {111} 的特征。如果不满足，说明这个链的滑移面不符合 FCC 位错的特征，直接跳过这个链，不将它纳入后续的螺旋链构建和分析中。

            // Screw character check
            int nfirst = snodes.front();//链上第一个节点的 id
            int nlast  = snodes.back();//链上最后一个节点的 id
            Vec3 pfirst = network->nodes[nfirst].pos;//获取链上第一个节点的位置信息，作为链的起点位置。后续会用它来计算链的方向向量。
            Vec3 plast  = network->cell.pbc_position(pfirst,
                                                     network->nodes[nlast].pos);//获取链上最后一个节点的位置信息，并考虑周期边界条件，得到 plast 作为链的终点位置。这样就得到了链的起点和终点位置，可以用它们来计算链的整体方向向量。
            Vec3 chain_dir = plast - pfirst;//计算链的方向向量，即从起点指向终点的向量。后续会用这个向量来判断链的螺旋特征，即它与 Burgers 向量的夹角是否足够小。
            double chain_len = chain_dir.norm();//计算链的长度，即方向向量的模长。后续会用这个长度来判断链是否足够长，以及在计算交滑移概率时作为链长参数。
            if (chain_len < tol) continue;//如果链的长度小于容差，说明这个链过短，无法进行可靠的螺旋特征判断和交滑移分析，因此跳过这个链，不将它纳入后续的螺旋链构建和分析中。
            chain_dir = chain_dir / chain_len;//将链的方向向量归一化，得到单位向量 chain_dir。后续会用这个单位向量来计算它与 Burgers 向量的夹角余弦值，以判断链的螺旋特征。 算链的整体方向（首到尾的弦向量）。pbc_position 把 nlast 的坐标在周期性边界下取离 pfirst 最近的镜像，避免跨边界时方向算错。

            Vec3 bhat = burg.normalized();//将 Burgers 向量归一化，得到单位向量 bhat。后续会用这个单位向量来计算它与链的方向向量 chain_dir 的夹角余弦值，以判断链的螺旋特征。螺旋段的定义是：段的方向与 Burgers 向量的夹角小于某个容差（比如 15 度）。计算夹角余弦值：cos(theta) = dot(chain_dir, bhat)，如果 cos(theta) 接近 1 或 -1，说明夹角接近 0 或 180 度，段具有强烈的螺旋特征；如果 cos(theta) 接近 0，说明夹角接近 90 度，段不具有螺旋特征。
            double screw_alignment = fabs(dot(chain_dir, bhat));//计算链的方向向量 chain_dir 与 Burgers 向量的单位向量 bhat 的点积，并取绝对值，得到 screw_alignment。这个值表示链的整体方向与 Burgers 向量的对齐程度，范围在 0 到 1 之间。screw_alignment 越接近 1，说明链越接近完全沿 Burgers 向量的方向，即具有强烈的螺旋特征；screw_alignment 越接近 0，说明链越垂直于 Burgers 向量，即不具有螺旋特征。后续会用这个值与螺旋角容差的余弦值进行比较，以判断链是否满足螺旋特征要求。

            int non_screw_count = 0;//统计链上不具有螺旋特征的段的数量。后续会用这个数量与链上段的总数进行比较，以判断链是否满足整体的螺旋特征要求。具体来说，如果链上超过一半的段不具有螺旋特征，说明这个链整体上不具有足够的螺旋特征，也会被跳过。
            for (int k = 0; k < nseg; k++) {
                int nk  = snodes[k];
                int nk1 = snodes[k+1];
                Vec3 pk  = network->nodes[nk].pos;
                Vec3 pk1 = network->cell.pbc_position(pk, network->nodes[nk1].pos);
                Vec3 seg_dir = pk1 - pk;
                double sl = seg_dir.norm();
                if (sl < tol) continue;
                if (fabs(dot(seg_dir / sl, bhat)) < scos) non_screw_count++;//对于链上每个段，计算它的方向向量 seg_dir，并归一化后与 Burgers 向量的单位向量 bhat 进行点积，得到这个段的对齐程度。如果这个值的绝对值小于螺旋角容差的余弦值 scos，说明这个段不具有足够的螺旋特征，统计到 non_screw_count 中。最后，如果 non_screw_count 超过链上段数的一半，说明这个链整体上不具有足够的螺旋特征，也会被跳过。
            }
            // ===== DEBUG: 螺型判定 (已注释；调试时取消注释即可) =====
            /*
            {
                double chord_ang = acos(fmin(1.0, screw_alignment)) * 180.0 / M_PI;
                bool rej = (screw_alignment < scos || non_screw_count > nseg / 2);
                double amin = 1e9, amax = -1e9;
                for (int kk = 0; kk < nseg; kk++) {
                    Vec3 pa = network->nodes[snodes[kk]].pos;
                    Vec3 pb = network->cell.pbc_position(pa, network->nodes[snodes[kk+1]].pos);
                    Vec3 d  = pb - pa; double dl = d.norm();
                    if (dl < tol) continue;
                    double a = acos(fmin(1.0, fabs(dot(d/dl, bhat)))) * 180.0 / M_PI;
                    amin = fmin(amin, a); amax = fmax(amax, a);
                }
                fprintf(stderr, "[OLD-CS] link=%d nseg=%d chord=%.1f  seg_ang=[%.1f,%.1f] "
                                "non_screw=%d/%d -> %s\n",
                        l, nseg, chord_ang, amin, amax, non_screw_count, nseg,
                        rej ? "REJECT(whole arm)" : "keep");
            }
            */
            // ===== DEBUG end =====
            if (screw_alignment < scos || non_screw_count > nseg / 2) continue;//如果链的整体对齐程度 screw_alignment 小于螺旋角容差的余弦值 scos，说明链的整体方向与 Burgers 向量的夹角过大，不满足螺旋特征要求，直接跳过这个链，不将它纳入后续的螺旋链构建和分析中。或者，如果链上不具有螺旋特征的段的数量 non_screw_count 超过链上段数的一半，说明这个链整体上不具有足够的螺旋特征，也直接跳过这个链，不将它纳入后续的螺旋链构建和分析中。

            // Classify mechanism type
            MechanismType mechanism = Bulk;
            if (network->conn[nfirst].num > 2 ||
                network->conn[nlast].num  > 2) {
                mechanism = Intersection;
            }//根据链的端点节点的连接数来判断交滑移机制类型。如果链的第一个节点 nfirst 或最后一个节点 nlast 的连接数 num 大于 2，说明这个链的端点处存在位错交点，属于 Intersection 机制；否则，说明这个链完全位于体积内，没有交点，属于 Bulk 机制。 机制判定：conn[node].num 是该节点的连接度。普通位错节点连 2 段；>2 说明端点是位错结，归为 Intersection。

            ScrewChain chain;//构建一个 ScrewChain 结构体实例，存储这个链的相关信息。包括链上节点 id 列表、段 id 列表、Burgers 向量、滑移面法向、交滑移机制类型等。后续会用这些信息来分析链的应力状态和计算交滑移概率。
            chain.node_ids.assign(snodes.begin(), snodes.end());
            chain.seg_ids.assign(ssegs.begin(), ssegs.end());
            chain.burg        = burg;
            chain.glide_plane = plane0;
            chain.mechanism   = mechanism;//将链上节点 id 列表和段 id 列表分别赋值给 chain 的 node_ids 和 seg_ids。将之前计算的 Burgers 向量 burg 和参考滑移面法向 plane0 分别赋值给 chain 的 burg 和 glide_plane。将之前判断的交滑移机制类型 mechanism 赋值给 chain 的 mechanism 字段。

            // For Intersection: identify junction arm and junction type
            if (mechanism == Intersection) {
                int jnode = (network->conn[nfirst].num > 2) ? nfirst : nlast;//对于 Intersection 机制的链，确定交点节点 jnode，即连接数大于 2 的那个端点节点。由于一个链的两个端点可能都连接数大于 2，这里优先选择 nfirst，如果 nfirst 的连接数大于 2 就选它，否则选 nlast。

                Vec3 jburg(0.0);//初始化交点处的合 Burgers 向量 jburg。后续会通过累加连接到交点节点 jnode 的段的 Burgers 向量来计算 jburg。注意在累加时需要排除链上已经包含的段，以避免重复计算。
                for (int k = 0; k < network->conn[jnode].num; k++) {
                    int sk = network->conn[jnode].seg[k];
                    bool in_chain = false;//检查这个段 sk 是否在当前链的段列表 ssegs 中。如果在，说明这个段已经被包含在链中，不应该再累加它的 Burgers 向量到 jburg 中；如果不在，说明这个段是连接到交点节点 jnode 的其他段，需要将它的 Burgers 向量根据连接顺序累加到 jburg 中。
                    for (int sc : ssegs) if (sc == sk) { in_chain = true; break; }//遍历链上段的 id 列表 ssegs，检查是否存在与当前段 sk 相同的 id。如果找到相同的 id，说明 sk 在链上，将 in_chain 置为 true，并跳出循环。否则，in_chain 保持为 false。
                    if (!in_chain) {
                        int ord = network->conn[jnode].order[k];
                        jburg = jburg + ord * network->segs[sk].burg;//如果 sk 不在链上，获取它在连接列表中的顺序 ord（可能是 +1 或 -1），根据 ord 的符号决定是否将 sk 的 Burgers 向量取反，然后累加到 jburg 中。这样就得到了交点处的合 Burgers 向量 jburg，它代表了交点处另一臂的 Burgers 向量。
                    }
                }
                chain.junction_burg = jburg;//将计算得到的交点处的合 Burgers 向量 jburg 赋值给 chain 的 junction_burg 字段。 计算结另一侧所有"非本链"臂的合 Burgers 矢量。order[k] 是该 segment 相对该节点的方向符号（±1），保证 b 加和方向正确。这个合矢量决定结的类型。

                for (int k = 0; k < network->conn[jnode].num; k++) {
                    int sk = network->conn[jnode].seg[k];
                    bool in_chain = false;//检查这个段 sk 是否在当前链的段列表 ssegs 中。如果在，说明这个段已经被包含在链中，不应该用它来判断结的滑移面；如果不在，说明这个段是连接到交点节点 jnode 的其他段，可以用它的滑移面来判断结的滑移面。由于一个交点可能连接多个段，这里只要找到一个不在链上的段，就可以用它的滑移面来判断结的滑移面。
                    for (int sc : ssegs) if (sc == sk) { in_chain = true; break; }//遍历链上段的 id 列表 ssegs，检查是否存在与当前段 sk 相同的 id。如果找到相同的 id，说明 sk 在链上，将 in_chain 置为 true，并跳出循环。否则，in_chain 保持为 false。
                    if (!in_chain) {
                        chain.junction_plane = network->segs[sk].plane;
                        break;//如果 sk 不在链上，获取它的滑移面法向量，并赋值给 chain 的 junction_plane 字段。然后跳出循环，因为只需要一个不在链上的段来判断结的滑移面。这个结的滑移面与结的 Burgers 向量一起决定了结的类型。
                    }
                }

                Vec3 jburg_crystal = system->crystal.Rinv * jburg.normalized();
                if (is_100_family(jburg_crystal)) {
                    chain.junction = Hirth;//如果交点处的合 Burgers 向量 jburg 在晶体坐标系下属于 ⟨100⟩ 家族，说明这个结是 Hirth 锁。
                } else if (is_110_family(jburg_crystal)) {
                    Vec3 jplane_crystal = system->crystal.Rinv
                                         * chain.junction_plane.normalized();
                    if (is_111_family(jplane_crystal)) {
                        chain.junction = GlideLock;//如果交点处的合 Burgers 向量 jburg 在晶体坐标系下属于 ⟨110⟩ 家族，并且结的滑移面在晶体坐标系下属于 {111} 家族，说明这个结是 Glide 锁。
                    } else if (is_110_family(jplane_crystal) || is_100_family(jplane_crystal)) {
                        chain.junction = LCLock;//如果交点处的合 Burgers 向量 jburg 在晶体坐标系下属于 ⟨110⟩ 家族，并且结的滑移面在晶体坐标系下属于 {110} 或 {100} 家族，说明这个结是 L-C 锁。
                    }
                } else {
                    chain.junction = UnknownLock;//如果交点处的合 Burgers 向量 jburg 在晶体坐标系下既不属于 ⟨100⟩ 家族也不属于 ⟨110⟩ 家族，说明这个结的类型无法确定，标记为 UnknownLock。
                }
            }

            chains.push_back(std::move(chain));//将构建好的 chain 添加到 chains 向量中，准备返回给调用者。使用 std::move 来避免不必要的复制，提高效率。 
        }
        return chains;//返回构建好的所有满足条件的螺旋链列表 chains。每个链都包含了它的节点和段信息、Burgers 向量、滑移面法向、交滑移机制类型，以及如果是 Intersection 机制还包含了结的 Burgers 向量和滑移面等信息。这些信息将用于后续的应力计算和交滑移分析。 按合 Burgers 矢量的族 + 面的族把结分成 Hirth / Glide lock / LC lock。std::move 把 chain 的内部 vector 资源转移进容器，省一次深拷贝。
    }

    // -----------------------------------------------------------------------
    //  Computes Schmid and Escaig stresses for a screw chain by averaging
    //  nodal forces over all segments.
    // -----------------------------------------------------------------------
    void compute_chain_stresses(SerialDisNet* network,
                                const ScrewChain& chain,
                                double burg_mag,
                                double& schmid_glide, double& schmid_cs,
                                double& escaig_glide, double& escaig_cs,
                                double& total_length)//计算链的 Schmid 应力和 Escaig 应力。首先根据链的 Burgers 向量和滑移面法向计算交滑移面法向，然后遍历链上的所有段，累加段两端节点的力，并计算链的总长度。最后将累积的力投影到滑移线和交滑移线方向上，得到平均的 Schmid 应力；同时根据 Burgers 向量、面法向和链长计算 Escaig 应力。注意这里的 Burgers 模长 burg_mag 是作为参数传入的，而不是直接用 chain.burg.norm()，这是为了避免重复计算和可能的数值误差。
    {
        Vec3 burg  = chain.burg;//获取链的 Burgers 向量
        Vec3 plane = chain.glide_plane.normalized();//获取链的滑移面法向量，并归一化。注意这里假设 chain.glide_plane 已经是一个有效的法向量，如果它的模长接近零，可能会导致数值问题。后续会用这个面法向来计算交滑移面法向和 Schmid 应力的方向。
        // Ensure consistent sign convention for Schmid stress calculation
        if (dot(plane, cross(burg, plane)) < 0.0) plane = -plane;//为了确保 Schmid 应力计算的一致符号约定，检查 plane 与 cross(burg, plane) 的点积是否小于零。如果小于零，说明 plane 的方向与 Burgers 向量和面法向的右手关系不一致，需要将 plane 取反。这样就保证了后续计算 Schmid 应力时，plane 的方向与 Burgers 向量和面法向的右手关系是一致的，从而得到正确的符号。 输出全部用引用参数返回。dot(plane, cross(burg,plane)) 用来固定面法向的正负号，保证 Schmid 应力符号一致。

        Vec3 cs_plane = get_crossslip_plane(plane, burg);//根据当前的滑移面法向 plane 和 Burgers 向量 burg 计算交滑移面法向 cs_plane。这个函数的实现之前已经分析过，它通过检查 Burgers 向量的分量来确定交滑移面法向的方向。注意如果 Burgers 向量不满足特定的特征（比如不是轴对齐的 1/2<110>），这个函数可能会返回零向量，这时后续的 Schmid 应力和 Escaig 应力计算将无法进行，需要特殊处理。
        if (cs_plane.norm() < 1e-6) {
            schmid_glide = schmid_cs = escaig_glide = escaig_cs = 0.0;
            total_length = 0.0;
            return;//如果计算得到的交滑移面法向 cs_plane 的模长小于容差，说明无法确定有效的交滑移面，这时将所有的 Schmid 应力和 Escaig 应力设置为零，并将链长 total_length 设置为零，然后返回。这样就避免了后续计算中使用无效的 cs_plane 导致的数值问题。
        }
        if (dot(cs_plane, cross(burg, cs_plane)) < 0.0) cs_plane = -cs_plane;//同样地，为了确保 Schmid 应力计算的一致符号约定，检查 cs_plane 与 cross(burg, cs_plane) 的点积是否小于零。如果小于零，说明 cs_plane 的方向与 Burgers 向量和交滑移面法向的右手关系不一致，需要将 cs_plane 取反。这样就保证了后续计算 Schmid 应力时，cs_plane 的方向与 Burgers 向量和交滑移面法向的右手关系是一致的，从而得到正确的符号。

        // Burgers direction for Escaig stress on each plane (Hussein et al. Sec. 2).
        Vec3 bhat = burg.normalized();
        Vec3 glide_bdir = bhat;
        Vec3 cs_bdir    = bhat;//根据 Hussein 等人论文中第 2 节的描述，Escaig 应力的 Burgers 方向对于滑移面和交滑移面都是沿 Burgers 向量的单位向量 bhat。也就是说，无论是计算滑移面的 Escaig 应力还是交滑移面的 Escaig 应力，Burgers 方向都是相同的，即 bhat。这是因为 Escaig 应力作用在刃分量上，而刃分量的方向与 Burgers 向量一致。 两个面上的Escaig的b方向都取b（螺型）

        Vec3 total_force(0.0);//初始化链上的总力为零向量。后续会通过遍历链上的所有段，累加段两端节点的力来计算这个总力。这个总力代表了作用在链上的整体力，可以用来计算平均的 Schmid 应力和 Escaig 应力。
        total_length = 0.0;//初始化链的总长度为零。后续会通过遍历链上的所有段，累加段的长度来计算这个总长度。这个总长度代表了链的实际长度，可以用来计算平均的 Schmid 应力和 Escaig 应力，以及在计算交滑移概率时作为链长参数。
        int nseg = (int)chain.seg_ids.size();
        for (int k = 0; k < nseg; k++) {
            int sk  = chain.seg_ids[k];
            int nk  = chain.node_ids[k];
            int nk1 = chain.node_ids[k+1];
            total_force  = total_force + 0.5 * (network->nodes[nk].f
                                               + network->nodes[nk1].f);//对于链上的每个段，获取它的两个端点节点 nk 和 nk1 的 id，然后获取这两个节点的力 vector，并取平均（乘以 0.5）后累加到 total_force 中。这样就得到了链上所有段两端节点的平均力的总和，代表了作用在链上的整体力。 注意这里是简单地对段两端节点的力取平均来近似段上的力分布，这在链较长且段较多时是合理的近似。
            total_length += network->seg_length(sk);//同时，获取每个段的长度，并累加到 total_length 中。这样就得到了链的实际长度，代表了链上所有段长度的总和。这个长度可以用来计算平均的 Schmid 应力和 Escaig 应力，以及在计算交滑移概率时作为链长参数。
        }//遍历链上的所有段，累加段两端节点的力来计算链上的总力 total_force，同时累加段的长度来计算链的总长度 total_length。 把每段两端节点力取平均再累加（0.5*(f_nk+f_nk1)），同时累加链总长。注意这样内部节点的力被计了两次半，等效于内节点全权重、端节点半权重。

        Vec3 glide_line = cross(burg, plane).normalized();//根据 Burgers 向量和滑移面法向计算滑移线方向 glide_line，并归一化。滑移线是 Burgers 向量和面法向的叉积方向，代表了位错在线上滑移的方向。后续会将 total_force 投影到这个方向上，得到平均的 Schmid 滑移应力。 glide面内的刃方向
        Vec3 cs_line    = cross(burg, cs_plane).normalized();//根据 Burgers 向量和交滑移面法向计算交滑移线方向 cs_line，并归一化。交滑移线是 Burgers 向量和交滑移面法向的叉积方向，代表了位错在线上交滑移的方向。后续会将 total_force 投影到这个方向上，得到平均的 Schmid 交滑移应力。交滑移面内的刃方向
        if (total_length > 1e-15) {
            schmid_glide = dot(total_force, glide_line) / total_length;
            schmid_cs    = dot(total_force, cs_line)    / total_length;
        } else {
            schmid_glide = schmid_cs = 0.0;
        }//如果链的总长度 total_length 大于容差，说明链具有有效的长度，可以进行 Schmid 应力的计算。将 total_force 分别投影到滑移线方向 glide_line 和交滑移线方向 cs_line 上，并除以 total_length，得到平均的 Schmid 滑移应力 schmid_glide 和平均的 Schmid 交滑移应力 schmid_cs。这样就得到了链上的平均 Schmid 应力，代表了作用在链上的整体应力状态。如果链的总长度小于容差，说明链过短，无法进行可靠的 Schmid 应力计算，将 schmid_glide 和 schmid_cs 设置为零。 Schmid 应力：合力投影到刃方向后除以总长。这里只除长度、没除 b，与 Escaig 不一致（量纲问题）。

        escaig_glide = compute_escaig_stress(total_force, glide_bdir, plane,    burg_mag, total_length);
        escaig_cs    = compute_escaig_stress(total_force, cs_bdir,    cs_plane, burg_mag, total_length);//根据 Hussein 等人论文中第 2 节的描述，计算链的平均 Escaig 滑移应力 escaig_glide 和平均 Escaig 交滑移应力 escaig_cs。这个计算需要 total_force、Burgers 方向、面法向、Burgers 模长和链长等参数。具体的计算公式在 compute_escaig_stress 函数中实现，通常涉及将 total_force 投影到 Burgers 方向上，并根据面法向和链长进行适当的归一化。这样就得到了链上的平均 Escaig 应力，代表了作用在链上的刃分量的应力状态。
    }

    // -----------------------------------------------------------------------
    //  Projects chain nodes onto the screw line and updates the glide plane.
    // -----------------------------------------------------------------------
    void execute_crossslip(SerialDisNet* network,
                           const ScrewChain& chain,
                           const Vec3& new_plane,
                           Mat33& dEp)//执行交滑移操作。首先根据链的端点节点的约束情况确定一个枢轴点 pivot，然后将链上每个节点的位置投影到新的滑移线上，更新节点位置以实现交滑移。最后更新链上每个段的滑移面法向为新的滑移面。这个函数直接修改网络中的节点位置和段信息，因此没有返回值。
    {
        if (chain.node_ids.empty()) return;//如果链上没有节点，说明这个链无效，直接返回，不执行任何操作。
        int nfirst = chain.node_ids.front();//获取链上第一个节点的 id，作为链的起点节点。后续会用这个节点的信息来确定枢轴点 pivot，以及在更新节点位置时进行特殊处理（如果这个节点受约束）。front() 是 std::vector 的成员函数，返回第一个元素的引用。
        int nlast  = chain.node_ids.back();//获取链上最后一个节点的 id，作为链的终点节点。后续会用这个节点的信息来确定枢轴点 pivot，以及在更新节点位置时进行特殊处理（如果这个节点受约束）。back() 是 std::vector 的成员函数，返回最后一个元素的引用。

        bool first_free = (network->nodes[nfirst].constraint == UNCONSTRAINED);//检查链的第一个节点 nfirst 是否是自由节点，即它的约束类型是否为 UNCONSTRAINED。如果是自由节点，说明它可以移动；如果不是自由节点，说明它受约束，不能移动。在后续更新节点位置时，如果这个节点受约束，将不会修改它的位置。这个信息对于确定枢轴点 pivot 和更新节点位置非常重要。
        bool last_free  = (network->nodes[nlast].constraint  == UNCONSTRAINED);//检查链的最后一个节点 nlast 是否是自由节点，即它的约束类型是否为 UNCONSTRAINED。如果是自由节点，说明它可以移动；如果不是自由节点，说明它受约束，不能移动。在后续更新节点位置时，如果这个节点受约束，将不会修改它的位置。这个信息对于确定枢轴点 pivot 和更新节点位置非常重要。

        Vec3 pivot(0.0);//初始化枢轴点 pivot 为零向量。后续会根据链的端点节点的约束情况来计算 pivot 的位置。pivot 是交滑移操作中的一个参考点，节点的位置将相对于这个点进行更新。
        Vec3 ref_pos = network->nodes[nfirst].pos;//获取链的第一个节点 nfirst 的位置，作为计算枢轴点 pivot 的参考位置 ref_pos。后续在计算 pivot 时会考虑周期边界条件，使得 pivot 的位置在 ref_pos 的附近。这个参考位置的选择是为了确保在更新节点位置时，能够正确处理周期边界条件，避免节点位置跨越边界时出现错误。
        if (first_free && last_free) {
            for (int nid : chain.node_ids){
                Vec3 p = network->cell.pbc_position(ref_pos, network->nodes[nid].pos);
                pivot = pivot + p;}//如果链的两个端点节点都是自由节点，说明链上的所有节点都可以移动。在这种情况下，计算枢轴点 pivot 的位置为链上所有节点位置的平均值。具体来说，遍历链上的每个节点 id nid，获取该节点的位置信息，并使用周期边界条件将其转换到以 ref_pos 为参考的最近镜像位置 p，然后将 p 累加到 pivot 中。最后将 pivot 除以链上节点的数量，得到所有节点位置的平均值作为枢轴点的位置。这样做的目的是为了在更新节点位置时，以链上所有节点的平均位置作为参考点，使得交滑移操作更加平滑和合理。
            pivot = pivot * (1.0 / (double)chain.node_ids.size());
        } else if (!first_free) {
            pivot = network->nodes[nfirst].pos;//如果链的第一个节点 nfirst 不是自由节点，说明它受约束，不能移动。在这种情况下，将枢轴点 pivot 的位置设置为第一个节点 nfirst 的位置。这样在更新节点位置时，以第一个节点的位置作为参考点，使得交滑移操作以这个受约束的节点为中心进行。 首端固定→绕首端
        } else {
            pivot = network->nodes[nlast].pos;//如果链的最后一个节点 nlast 不是自由节点，说明它受约束，不能移动。在这种情况下，将枢轴点 pivot 的位置设置为最后一个节点 nlast 的位置。这样在更新节点位置时，以最后一个节点的位置作为参考点，使得交滑移操作以这个受约束的节点为中心进行。 尾端固定→绕尾端
        }//根据链的端点节点的约束情况，计算枢轴点 pivot 的位置。如果链的两个端点节点都是自由节点，pivot 是链上所有节点位置的平均值；如果第一个节点受约束，pivot 是第一个节点的位置；如果最后一个节点受约束，pivot 是最后一个节点的位置。这个 pivot 将作为后续更新节点位置时的参考点。 

        Vec3 bhat = chain.burg.normalized();//获取链的 Burgers 向量，并归一化得到单位向量 bhat。这个 bhat 将用于计算节点位置的投影方向，以及在更新节点位置时保持 Burgers 向量的方向不变。交滑移操作应该保持 Burgers 向量不变，因此节点的位置更新将沿着以 pivot 为中心、bhat 方向的线进行。
        for (int i = 0; i < (int)chain.node_ids.size(); i++) {
            int  nid         = chain.node_ids[i];
            bool is_endpoint = (i == 0 || i == (int)chain.node_ids.size() - 1);//对于链上的每个节点，获取它的 id nid，并判断它是否是链的端点节点。端点节点是指链上的第一个节点（i == 0）或最后一个节点（i == size - 1）。这个信息在更新节点位置时非常重要，因为如果端点节点受约束，它们的位置将不会被修改。
            if (is_endpoint) {
                bool moveable = (i == 0) ? first_free : last_free;
                if (!moveable) continue;
            }//如果当前节点是端点节点，检查它是否是可移动的。如果是第一个节点，检查 first_free；如果是最后一个节点，检查 last_free。如果端点节点不可移动，说明它受约束，不能修改位置，直接跳过这个节点，不进行更新。这样就保证了受约束的端点节点的位置保持不变。
            Vec3 pos  = network->cell.pbc_position(pivot, network->nodes[nid].pos);//获取当前节点的位置信息，并使用周期边界条件将其转换到以 pivot 为参考的最近镜像位置 pos。这样做是为了确保在计算节点位置的投影时，能够正确处理周期边界条件，避免节点位置跨越边界时出现错误。pos 是当前节点在 pivot 附近的有效位置。
            Vec3 proj = pivot + dot(pos - pivot, bhat) * bhat;//将节点的位置 pos 投影到以 pivot 为中心、bhat 方向的线上，得到投影位置 proj。具体来说，计算 pos 与 pivot 的向量差，并将其在 bhat 方向上的分量（通过点积计算）乘以 bhat，得到从 pivot 到 proj 的向量，然后加上 pivot 得到 proj 的位置。这样就实现了将节点位置投影到交滑移线上的操作。proj 是节点在交滑移线上的新位置。 投影到过pivot、沿b的直线
            network->move_node(nid, proj, dEp);//调用网络的 move_node 函数，将节点 nid 移动到新的位置 proj，并传入 dEp 以记录位移引起的塑性应变增量。这个操作直接修改了网络中节点的位置，实现了交滑移操作中节点位置的更新。通过将节点位置投影到交滑移线上，保持了 Burgers 向量的方向不变，同时实现了链的交滑移。
        }//遍历链上的每个节点，计算它的新位置 proj，并调用 network->move_node 来更新节点的位置。对于端点节点，如果它们受约束，则跳过不更新位置。对于非端点节点，直接更新位置。这样就完成了链上所有节点的交滑移位置更新。 把每个节点投影到"过 pivot、方向为 b"的螺型直线上。这条线正是两张 {111} 面的交线，所以投影后位错就成了一条理想螺型，可以放到任一面上。move_node 同时把节点移动产生的塑性应变增量累加到 dEp。

        for (int sk : chain.seg_ids)
            update_seg_plane(network, sk, new_plane);//遍历链上的每个段 id sk，调用 update_seg_plane 来更新段的滑移面法向为 new_plane。这个函数会直接修改网络中段的信息，确保链上所有段的滑移面法向都被更新为新的滑移面，从而完成整个链的交滑移操作。 把所有段的滑移面改成交滑移面
    }

    // -----------------------------------------------------------------------
    //  Bulk cross-slip: thermally activated, uses actual chain length.
    // -----------------------------------------------------------------------
    void handle_bulk(System* system, SerialDisNet* network,
                     const ScrewChain& chain)//处理 Bulk 机制的交滑移。首先计算链的 Schmid 应力和 Escaig 应力，然后根据 Hussein 等人论文中第 2 节的描述，检查链是否满足交滑移的应力条件。如果满足条件，再根据链的实际长度和系统参数计算交滑移的热激活概率，并进行随机抽样决定是否执行交滑移。最后，如果决定执行交滑移，计算新的交滑移面并调用 execute_crossslip 来更新网络中的节点位置和段信息。这个函数实现了 Bulk 机制下的交滑移分析和操作。
    {
        double burg_mag = system->params.burgmag;
        double schmid_glide, schmid_cs, escaig_glide, escaig_cs, total_length;
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_glide, schmid_cs,
                               escaig_glide, escaig_cs, total_length);
        if (total_length < 1e-15) return;//如果链的总长度 total_length 小于容差，说明链过短，无法进行可靠的应力计算和交滑移分析，直接返回，不执行任何操作。

        // Stress threshold conditions (Hussein et al. Sec. 2):
        //   1. |tau_cs| >= mu*b / (10*L)
        //   2. |tau_cs| >= 1.1|tau_glide|  (prevents cross-slip back and forth)
        double stress_threshold = system->params.MU * system->params.burgmag
                                  / (10.0 * total_length);//根据 Hussein 等人论文中第 2 节的描述，计算交滑移的应力阈值 stress_threshold。这个阈值是根据材料的剪切模量 MU、Burgers 模长 burgmag 和链的总长度 total_length 计算得到的，具体公式为 MU * b / (10 * L)。这个阈值用于判断链上的 Schmid 交滑移应力是否足够大，以满足交滑移发生的条件。注意这里使用了链的实际长度 total_length 来计算应力阈值，这也是 Bulk 机制的特点之一。
        if (fabs(schmid_cs) < stress_threshold)   return;//根据 Hussein 等人论文中第 2 节的描述，首先检查链上的 Schmid 交滑移应力 schmid_cs 的绝对值是否小于应力阈值 stress_threshold。如果小于阈值，说明链上的交滑移应力不足以驱动交滑移发生，直接返回，不执行交滑移操作。 条件1：cs面应力够大
        if (fabs(schmid_cs) < 1.1*fabs(schmid_glide)) return;//根据 Hussein 等人论文中第 2 节的描述，接着检查链上的 Schmid 交滑移应力 schmid_cs 的绝对值是否小于 1.1 倍的 Schmid 滑移应力 schmid_glide 的绝对值。如果小于这个条件，说明链上的交滑移应力不足以克服滑移应力的影响，可能会导致交滑移发生后立即反向滑移回原来的面，因此直接返回，不执行交滑移操作。这个条件是为了防止交滑移发生后立即反向滑移回原来的面，从而实现更稳定的交滑移行为。 条件2：cs面比glide面大10%以上

        const Params& p = params;//为了代码简洁，将系统参数 params 引用为 p，方便后续使用。这个 p 包含了交滑移分析和操作所需的各种参数，如激活能、激活体积、尝试频率、温度等。
        double V_a = p.bulkActivationVolume * burg_mag * burg_mag * burg_mag;//根据系统参数中的 bulkActivationVolume 和 Burgers 模长 burg_mag 计算激活体积 V_a。通常情况下，激活体积是以 Burgers 向量的立方为单位的，因此这里将 bulkActivationVolume 乘以 burg_mag 的三次方来得到实际的激活体积。这个 V_a 将用于计算交滑移的热激活概率。
        double dSigma_E  = escaig_glide - escaig_cs;
        double dt        = p.evalFrequency * system->params.nextdt;//根据系统参数中的 evalFrequency 和 nextdt 计算时间增量 dt。这个 dt 代表了交滑移事件的评估频率与系统时间步长的乘积，表示在多长的时间间隔内评估一次交滑移事件。这个 dt 将用于计算交滑移的热激活概率。
        double prob      = compute_thermal_probability(p.bulkActivationEnergy, V_a, dSigma_E,
                                                      p.temperature,
                                                      p.bulkAttemptFrequency, dt,
                                                      total_length,
                                                      p.bulkReferenceLength);//根据系统参数中的 bulkActivationEnergy、激活体积 V_a、能量差 dSigma_E、温度 temperature、尝试频率 bulkAttemptFrequency、时间增量 dt、链的总长度 total_length 和参考长度 bulkReferenceLength，计算交滑移的热激活概率 prob。这个函数 compute_thermal_probability 会根据这些参数使用适当的物理模型（如 Arrhenius 公式）来计算交滑移发生的概率。这个 prob 将用于后续的随机抽样，以决定是否执行交滑移操作。
        if (prob < 1.0 && uniform01(rng) > prob) return;//如果计算得到的交滑移热激活概率 prob 小于 1.0，说明交滑移发生不是必然的，而是具有一定的随机性。在这种情况下，生成一个 [0, 1) 范围内的随机数，如果这个随机数大于 prob，说明这次评估没有触发交滑移事件，直接返回，不执行交滑移操作。只有当随机数小于或等于 prob 时，才会继续执行交滑移操作。这样就实现了基于热激活概率的随机抽样机制，使得交滑移事件的发生具有统计意义。 掷骰子；prob≥1则必发生

        Vec3 plane = chain.glide_plane.normalized();//如果决定执行交滑移，首先获取链的当前滑移面法向，并归一化。这个 plane 将用于计算新的交滑移面法向。注意这里假设 chain.glide_plane 已经是一个有效的法向量，如果它的模长接近零，可能会导致数值问题。后续会用这个面法向来计算交滑移面法向和执行交滑移操作。
        if (dot(plane, cross(chain.burg, plane)) < 0.0) plane = -plane;//为了确保 Schmid 应力计算的一致符号约定，检查 plane 与 cross(chain.burg, plane) 的点积是否小于零。如果小于零，说明 plane 的方向与 Burgers 向量和面法向的右手关系不一致，需要将 plane 取反。这样就保证了后续计算 Schmid 应力时，plane 的方向与 Burgers 向量和面法向的右手关系是一致的，从而得到正确的符号。 输出全部用引用参数返回。dot(plane, cross(burg,plane)) 用来固定面法向的正负号，保证 Schmid 应力符号一致。
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);//根据当前的滑移面法向 plane 和 Burgers 向量 chain.burg 计算新的交滑移面法向 cs_plane。这个函数的实现之前已经分析过，它通过检查 Burgers 向量的分量来确定交滑移面法向的方向。注意如果 Burgers 向量不满足特定的特征（比如不是轴对齐的 1/2<110>），这个函数可能会返回零向量，这时后续的 Schmid 应力和 Escaig 应力计算将无法进行，需要特殊处理。
        if (cs_plane.norm() < 1e-6) return;//如果计算得到的新的交滑移面法向 cs_plane 的模长小于容差，说明无法确定有效的交滑移面，这时直接返回，不执行交滑移操作。这样就避免了后续计算中使用无效的 cs_plane 导致的数值问题。
        execute_crossslip(network, chain, cs_plane, system->dEp);//调用 execute_crossslip 来执行交滑移操作，传入网络、链、计算得到的新的交滑移面法向 cs_plane，以及系统的塑性应变增量 dEp。这个函数会更新网络中链上节点的位置和段的信息，实现交滑移操作。这样就完成了 Bulk 机制下的交滑移分析和操作。
    }

    // -----------------------------------------------------------------------
    //  Intersection cross-slip:
    //    Repulsive (spontaneous): executes immediately without thermal check.
    //    Attractive (thermally activated): uses fixed effective length and
    //    lock-type specific activation parameters.
    // -----------------------------------------------------------------------
    void handle_intersection(System* system, SerialDisNet* network,
                             const ScrewChain& chain)//处理 Intersection 机制的交滑移。首先计算链的 Schmid 应力和 Escaig 应力，然后根据 Hussein 等人论文中第 2 节的描述，检查链是否满足交滑移的应力条件。如果满足条件，再根据链的实际长度和系统参数计算交滑移的热激活概率，并进行随机抽样决定是否执行交滑移。对于具有特定交滑移锁类型（如 Hirth 锁、Glide 锁、LC 锁）的链，如果它们满足条件，将直接执行交滑移而不进行热激活检查；对于其他类型的链，则按照热激活机制进行处理。最后，如果决定执行交滑移，计算新的交滑移面并调用 execute_crossslip 来更新网络中的节点位置和段信息。这个函数实现了 Intersection 机制下的交滑移分析和操作。
    {
        const double tol = 1e-6;
        Vec3 plane = chain.glide_plane.normalized();
        if (dot(plane, cross(chain.burg, plane)) < 0.0) plane = -plane;
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);//计算交滑移面法向 cs_plane，并进行符号约定处理。这个 cs_plane 将用于后续的交滑移分析和操作。注意如果 Burgers 向量不满足特定的特征（比如不是轴对齐的 1/2<110>），这个函数可能会返回零向量，这时后续的 Schmid 应力和 Escaig 应力计算将无法进行，需要特殊处理。
        if (cs_plane.norm() < tol) return;//如果计算得到的交滑移面法向 cs_plane 的模长小于容差 tol，说明无法确定有效的交滑移面，这时直接返回，不执行交滑移操作。这样就避免了后续计算中使用无效的 cs_plane 导致的数值问题。

        // Repulsive intersection (Hussein et al. Sec. 2): spontaneous and athermal
        // when junction Burgers vector sum is <112> type.
        Vec3 jburg_crystal = system->crystal.Rinv * chain.junction_burg.normalized();
        if (is_112_family(jburg_crystal)) {
            execute_crossslip(network, chain, cs_plane, system->dEp);
            return;//根据 Hussein 等人论文中第 2 节的描述，如果链的交滑移锁类型是排斥性交滑移（Repulsive intersection），即当链的交滑移锁的 Burgers 向量和属于 <112> 家族时，说明这种交滑移是自发的、非热激活的，不需要进行热激活检查，直接执行交滑移操作。首先将链的交滑移锁的 Burgers 向量转换到晶体坐标系下，并归一化得到 jburg_crystal，然后检查 jburg_crystal 是否属于 <112> 家族。如果是，则直接调用 execute_crossslip 来执行交滑移操作，并传入网络、链、计算得到的交滑移面法向 cs_plane，以及系统的塑性应变增量 dEp。执行完后直接返回，不进行后续的热激活分析。
        }

        // Attractive: thermally activated, parameters depend on junction type
        double burg_mag = system->params.burgmag;
        const Params& p = params;
        double Ea, Va, freq, L_ref, eff_len;//根据链的交滑移锁类型，设置不同的激活能 Ea、激活体积 Va、尝试频率 freq、参考长度 L_ref 和有效长度 eff_len。对于具有特定交滑移锁类型（如 Hirth 锁、Glide 锁、LC 锁）的链，使用对应的系统参数来设置这些值；对于其他类型的链，则使用 bulk 参数作为默认值。这个设置将用于后续的热激活概率计算，以决定是否执行交滑移操作。

        if (chain.junction == Hirth) {
            Ea      = p.hirthActivationEnergy;
            Va      = p.hirthActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.hirthAttemptFrequency;
            L_ref   = p.hirthReferenceLength;
            eff_len = p.hirthEffectiveLength / burg_mag;
        } else if (chain.junction == GlideLock) {
            Ea      = p.glideLockActivationEnergy;
            Va      = p.glideLockActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.glideLockAttemptFrequency;
            L_ref   = p.glideLockReferenceLength;
            eff_len = p.glideLockEffectiveLength / burg_mag;
        } else if (chain.junction == LCLock) {
            Ea      = p.lcLockActivationEnergy;
            Va      = p.lcLockActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.lcLockAttemptFrequency;
            L_ref   = p.lcLockReferenceLength;
            eff_len = p.lcLockEffectiveLength / burg_mag;
        } else {
            // Unknown junction type: fall back to bulk parameters
            Ea      = p.bulkActivationEnergy;
            Va      = p.bulkActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.bulkAttemptFrequency;
            L_ref   = p.bulkReferenceLength;
            eff_len = -1.0; // use actual chain length
        }

        double schmid_g, schmid_c, escaig_g, escaig_c, total_length;//计算链的 Schmid 应力和 Escaig 应力，以及链的总长度 total_length。这个计算将使用之前分析过的 compute_chain_stresses 函数，根据链上的节点力、Burgers 向量、滑移面法向和交滑移面法向来计算这些应力值。计算得到的 schmid_g、schmid_c、escaig_g 和 escaig_c 将用于后续的热激活概率计算，以决定是否执行交滑移操作。
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_g, schmid_c, escaig_g, escaig_c,
                               total_length);

        double dSigma_E  = escaig_g - escaig_c;
        double dt        = p.evalFrequency * system->params.nextdt;
        double use_len   = (eff_len > 0.0) ? eff_len : total_length;
        double prob      = compute_thermal_probability(Ea, Va, dSigma_E,
                                                      p.temperature,
                                                      freq, dt, use_len, L_ref);
        if (prob < 1.0 && uniform01(rng) > prob) return;//如果计算得到的交滑移热激活概率 prob 小于 1.0，说明交滑移发生不是必然的，而是具有一定的随机性。在这种情况下，生成一个 [0, 1) 范围内的随机数，如果这个随机数大于 prob，说明这次评估没有触发交滑移事件，直接返回，不执行交滑移操作。只有当随机数小于或等于 prob 时，才会继续执行交滑移操作。这样就实现了基于热激活概率的随机抽样机制，使得交滑移事件的发生具有统计意义。 掷骰子；prob≥1则必发生

        execute_crossslip(network, chain, cs_plane, system->dEp);
    }

public:
    CrossSlipFCCThermal() = default;
    CrossSlipFCCThermal(System* /*system*/, Force* _force, Params _params)
        : force(_force), params(_params),
          rng(std::random_device{}()) {}//构造函数，接受系统指针、力对象指针和参数对象作为输入，并初始化成员变量。force 用于计算节点上的力，params 包含了交滑移分析和操作所需的各种参数，rng 是一个随机数生成器，用于在热激活机制中进行随机抽样。这个构造函数没有使用系统指针，但保留了这个参数以保持与其他处理器类的一致接口。

    void handle(System* system) override//重写基类的 handle 函数，实现交滑移分析和操作的核心逻辑。首先进行 Kokkos fencing 来确保之前的计算完成，然后检查晶体类型和滑移面使用情况是否满足要求。接着根据评估频率决定是否进行交滑移分析，如果需要分析，则获取网络对象，构建螺链，并针对每个链根据其机制（Bulk 或 Intersection）调用相应的处理函数来执行交滑移分析和操作。最后再次进行 Kokkos fencing 来确保交滑移操作完成，并停止计时器。这个函数是整个 CrossSlipFCCThermal 类的核心，负责协调和执行交滑移相关的计算和操作。
    {
        Kokkos::fence();//在处理交滑移之前，调用 Kokkos::fence() 来确保之前的所有 Kokkos 计算都已经完成。这是为了保证在进行交滑移分析和操作时，所有相关的数据都是最新的，并且没有未完成的计算可能会影响结果。Kokkos::fence() 是 Kokkos 库提供的一个函数，用于同步设备上的计算，确保所有之前提交的计算都已经完成。
        system->timer[system->TIMER_CROSSSLIP].start();//启动交滑移处理的计时器，以便在性能分析中记录交滑移处理的时间。这个计时器会在整个 handle 函数执行期间运行，直到最后停止。通过这个计时器，可以评估交滑移处理的性能，并与其他处理器的性能进行比较。

        if (system->crystal.type != FCC_CRYSTAL)
            ExaDiS_fatal("Error: CrossSlipFCCThermal requires FCC_CRYSTAL\n");//首先检查系统的晶体类型是否为 FCC_CRYSTAL。如果不是 FCC 晶体，说明这个交滑移处理器不适用于当前系统，直接报错并终止程序。这个检查是为了确保交滑移分析和操作的前提条件得到满足，因为这个处理器的算法是基于 FCC 晶体结构的特性设计的。
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: CrossSlipFCCThermal requires use_glide_planes=true\n");//接着检查系统的晶体参数中是否启用了滑移面使用（use_glide_planes）。如果没有启用，说明这个交滑移处理器无法正确执行，因为它依赖于滑移面信息来进行分析和操作，直接报错并终止程序。这个检查是为了确保系统的配置满足交滑移处理器的要求。

        eval_counter++;//增加评估计数器 eval_counter 的值，用于跟踪已经执行了多少次交滑移评估。这个计数器在每次调用 handle 函数时都会增加，主要用于根据系统参数中的 evalFrequency 来决定是否进行交滑移分析。如果 evalFrequency 大于 1，则只有当 eval_counter 达到 evalFrequency 的倍数时才会进行交滑移分析，否则直接返回。这种机制可以用来控制交滑移分析的频率，减少计算开销。
        if (params.evalFrequency > 1 &&
            (eval_counter % params.evalFrequency) != 0) {
            system->timer[system->TIMER_CROSSSLIP].stop();
            return;
        }//根据系统参数中的 evalFrequency 和当前的 eval_counter 来决定是否进行交滑移分析。如果 evalFrequency 大于 1，说明不是每次调用 handle 都需要进行交滑移分析，而是需要根据 eval_counter 的值来控制频率。具体来说，只有当 eval_counter 是 evalFrequency 的倍数时，才会继续执行交滑移分析；否则直接停止计时器并返回，不执行交滑移操作。这样就实现了对交滑移分析频率的控制，可以减少计算开销，特别是在不需要每个时间步都进行交滑移分析的情况下。

        SerialDisNet* network = system->get_serial_network();//获取系统中的 SerialDisNet 网络对象，作为后续交滑移分析和操作的基础数据结构。这个网络对象包含了系统中所有节点、段和链的信息，以及它们之间的关系。通过这个网络对象，可以访问链上的节点力、Burgers 向量、滑移面法向等信息，并且可以调用函数来更新节点位置和段信息，实现交滑移操作。

        std::vector<ScrewChain> chains = build_screw_chains(system, network);//调用 build_screw_chains 函数来构建系统中的螺链（ScrewChain）。这个函数会根据网络中的节点和段信息，识别出所有满足条件的螺链，并将它们存储在一个 std::vector 中返回。每个 ScrewChain 包含了链上节点的 id、段的 id、Burgers 向量、滑移面法向、交滑移锁类型等信息，这些信息将用于后续的交滑移分析和操作。

        for (const ScrewChain& chain : chains) {
            switch (chain.mechanism) {
                case Bulk:         handle_bulk(system, network, chain);         break;
                case Intersection: handle_intersection(system, network, chain); break;
            }//遍历构建好的螺链，根据每个链的机制类型（Bulk 或 Intersection）调用相应的处理函数来执行交滑移分析和操作。对于 Bulk 机制的链，调用 handle_bulk；对于 Intersection 机制的链，调用 handle_intersection。这样就根据不同的交滑移机制对每个链进行分析和操作，实现了整个系统中所有满足条件的链的交滑移处理。
        }

        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].stop();//在完成所有链的交滑移处理后，再次调用 Kokkos::fence() 来确保所有的交滑移操作都已经完成，然后停止计时器。这样就完整地记录了整个交滑移处理的时间，并确保在性能分析中得到准确的结果。
    }

    const char* name() override { return "CrossSlipFCCThermal"; }
};


/*---------------------------------------------------------------------------
 *
 *    Class:    CrossSlipFCCThermalParallel<F>
 *              Kokkos-aware wrapper for CrossSlipFCCThermal.
 *              Adds Kokkos fencing around the serial algorithm for GPU builds.
 *
 *-------------------------------------------------------------------------*/
template<class F>//定义一个模板类 CrossSlipFCCThermalParallel，继承自 CrossSlipFCCThermal。这个类是一个 Kokkos-aware 的包装器，用于在 GPU 构建中添加 Kokkos fencing 来确保计算的正确性。通过在 handle 函数中添加 Kokkos fencing，可以确保在执行交滑移分析和操作时，所有相关的计算都已经完成，从而避免数据竞争和不一致的问题。这个类的实现非常简单，主要是在 handle 函数的开始和结束处添加 Kokkos fencing，然后调用基类的 handle 函数来执行实际的交滑移处理逻辑。
class CrossSlipFCCThermalParallel : public CrossSlipFCCThermal {
public:
    CrossSlipFCCThermalParallel(System* system, Force* _force,
                                CrossSlipFCCThermal::Params _params)
        : CrossSlipFCCThermal(system, _force, _params) {}//构造函数，接受系统指针、力对象指针和参数对象作为输入，并调用基类的构造函数来初始化成员变量。这个构造函数没有添加额外的逻辑，只是简单地将参数传递给基类的构造函数。

    void handle(System* system) override//重写基类的 handle 函数，在函数开始和结束处添加 Kokkos fencing 来确保计算的正确性。首先调用 Kokkos::fence() 来确保之前的计算完成，然后调用基类的 handle 函数来执行实际的交滑移处理逻辑，最后再次调用 Kokkos::fence() 来确保交滑移操作完成。通过这种方式，可以在 GPU 构建中确保数据的一致性和正确性，避免由于并行计算引起的数据竞争和不一致问题。
    {
        Kokkos::fence();
        CrossSlipFCCThermal::handle(system);
        Kokkos::fence();
    }//在 handle 函数的开始和结束处添加 Kokkos fencing 来确保计算的正确性。通过调用 Kokkos::fence()，可以确保在执行交滑移分析和操作时，所有相关的计算都已经完成，从而避免数据竞争和不一致的问题。这个函数主要是为了在 GPU 构建中使用 Kokkos fencing 来保护基类的 handle 函数的执行。

    const char* name() override { return "CrossSlipFCCThermalParallel"; }//重写 name 函数，返回类的名称 "CrossSlipFCCThermalParallel"。这个函数可以用于在日志或性能分析中标识当前使用的交滑移处理器的类型。通过返回不同的名称，可以区分使用了 Kokkos fencing 的并行版本和没有 fencing 的串行版本，从而在性能分析中更清晰地了解不同处理器的表现。
};//重写 name 函数，返回类的名称 "CrossSlipFCCThermalParallel"。这个函数可以用于在日志或性能分析中标识当前使用的交滑移处理器的类型。

} // namespace ExaDiS

#endif // EXADIS_CROSS_SLIP_FCC_THERMAL_H
