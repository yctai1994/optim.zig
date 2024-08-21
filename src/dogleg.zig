//! References
//! [1] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Lemma 4.2

const Errors = error{DoglegError};

// Bk := approx. Hessian at xk, should be Cholesky factorized (Rᵀ⋅R)
// gk := gradient at xk
// pk := final search direction for xk
// pN := buffer for quasi-Newton direction
// pS := buffer for steepest-descent direction
// sz := trust-region max. step size Δₖ
fn dogleg(Bk: [][]f64, gk: []f64, pk: []f64, pN: []f64, pS: []f64, sz: f64) !void {
    const n: usize = Bk.len;
    if (n != gk.len or n != pN.len or n != pS.len) return error.DoglegError;

    const sz2: f64 = sz * sz;

    // pN ← -Bk⁻¹⋅∇f(xk)
    for (0..n) |i| pk[i] = -gk[i]; // directly use pk for possibly quick return
    try trsv('R', 'T', Bk, pk);
    try trsv('R', 'N', Bk, pk);

    // check ‖pN‖² ≤ Δₖ²
    const norm2_pN: f64 = dot(pk, pk);
    if (norm2_pN <= sz2) return else @memcpy(pN, pk);

    // pS ← -αₖ⋅∇f(xk)
    const norm2_gk: f64 = dot(gk, gk); // ‖gk‖²
    const trust_ak: f64 = sz / @sqrt(norm2_gk); // Δₖ/‖gk‖
    // gkᵀ⋅Bk⋅gk = gkᵀ⋅(Lk⋅Rk)⋅gk
    @memcpy(pk, gk); // here, pk is just a temp. buffer
    try trmv('R', 'N', Bk, pk);
    const steep_ak: f64 = @min(trust_ak, norm2_gk / dot(pk, pk)); // min( Δₖ/‖gk‖,  ‖gk‖²/gkᵀ⋅Bk⋅gk )
    for (0..n) |i| pS[i] = -steep_ak * gk[i]; // pS ← -αₖ⋅∇f(xk)

    // ‖pS + β⋅(pN - pS)‖² = Δₖ², s.t. 0 ≤ β ≤ 1
    for (0..n) |i| pk[i] = pN[i] - pS[i];
    const a: f64 = dot(pk, pk); // ‖pN - pS‖²
    const b: f64 = dot(pS, pk); // pS⋅(pN - pS)
    const c: f64 = dot(pS, pS) - sz2; // ‖pS‖² - Δₖ²
    const d: f64 = @sqrt(b * b - a * c); // b² - ac
    const s: f64 = (-b - d) / a; // smaller root
    const t: f64 = (-b + d) / a; // larger root

    // pk = pS + β⋅(pN - pS)
    if (0.0 < t and t <= 1.0) { // attempt to use the larger one
        for (0..n) |i| pk[i] = pS[i] + t * pk[i];
    } else if (0.0 < s and s <= 1.0) {
        for (0..n) |i| pk[i] = pS[i] + s * pk[i];
    } else {
        @memcpy(pk, pS);
    }

    return;
}

fn dot(x: []f64, y: []f64) f64 {
    var t: f64 = 0.0;
    for (x, y) |x_i, y_i| t += x_i * y_i;
    return t;
}

const trmv = @import("./linalg/trmv.zig").trmv;
const trsv = @import("./linalg/trsv.zig").trsv;
