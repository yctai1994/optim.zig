//! References
//! [1] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Lemma 4.2

const Errors = error{DimensionMismatch};

// Bk := Hessian at xk, should be Cholesky factorized (Rᵀ⋅R)
// gk := gradient at xk
// pk := dogleg's step for xk
// pN := buffer for quasi-Newton's step
// pS := buffer for steepest-descent's step
// sz := trust-region max. step size Δₖ
fn dogleg(Bk: [][]f64, gk: []f64, pk: []f64, pN: []f64, pS: []f64, sz: f64) !void {
    const n: usize = Bk.len;
    if (n != gk.len or n != pN.len or n != pS.len) return error.DimensionMismatch;

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
    for (0..n) |i| pk[i] = pN[i] - pS[i]; // here, pk is just a temp. buffer
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

test "Rosenbrock: xk = .{ -1.2, 1.0 }" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    // Bk = Rᵀ⋅R = cholescky( exact Hessian at xk = .{ -1.2, 1.0 } )
    const Bk: [][]f64 = try ArrF64.matrix(2, 2);
    defer ArrF64.free(Bk);

    inline for (.{ 0x1.23c0d99c17436p+5, 0x1.a52d7f6fc5311p+3 }, Bk[0]) |v, *p| p.* = v;
    inline for (.{ 0x1.0000000000000p+0, 0x1.4b1d7f7c3508bp+2 }, Bk[1]) |v, *p| p.* = v;

    // gk = exact gradient at xk = .{ -1.2, 1.0 }
    const gk: []f64 = try ArrF64.vector(2);
    defer ArrF64.free(gk);

    inline for (.{ -0x1.af33333333332p+7, -0x1.5ffffffffffffp+6 }, gk) |v, *p| p.* = v;

    const pk: []f64 = try ArrF64.vector(2);
    defer ArrF64.free(pk);

    const pN: []f64 = try ArrF64.vector(2);
    defer ArrF64.free(pN);

    const pS: []f64 = try ArrF64.vector(2);
    defer ArrF64.free(pS);

    // trust-region max. step size at xk = .{ -1.2, 1.0 }
    const sz: f64 = 3.0e-1;

    try dogleg(Bk, gk, pk, pN, pS, sz);
    try testing.expect(@sqrt(dot(pk, pk)) <= sz);
}

fn dot(x: []f64, y: []f64) f64 {
    var t: f64 = 0.0;
    for (x, y) |x_i, y_i| t += x_i * y_i;
    return t;
}

const std = @import("std");
const testing = std.testing;

const trmv = @import("./linalg/trmv.zig").trmv;
const trsv = @import("./linalg/trsv.zig").trsv;
const Array = @import("./linalg/array.zig").Array;
