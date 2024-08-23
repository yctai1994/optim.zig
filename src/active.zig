//! References
//! [1] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Sec. 16.7

const Errors = error{DimensionMismatch};

// t := buffer, storing max. step of each dimension
// x := start point
// d := direction (including length)
fn checkbounds(lb: []f64, ub: []f64, x: []f64, d: []f64, t: []f64) void {
    const n: usize = x.len;
    for (0..n) |i| {
        if (-INF < lb[i] and d[i] < 0.0) {
            t[i] = (lb[i] - x[i]) / d[i];
        } else if (ub[i] < INF and 0.0 < d[i]) {
            t[i] = (ub[i] - x[i]) / d[i];
        } else { // (d_i == 0) or (no bounds in d_i direction)
            t[i] = INF;
        }
    }
}

// Use gradient projection to
// 1. check the working set.
// 2. get the Cauchy point
//
// tk := breakpoints' step sizes
// ts := buffer for sorted ts
// xt := buffer for piecewise line segments
// xc := buffer for storing Cauchy point
// Wk := buffer for working (active) set
// ta := pointer for counting number of activated bounds
fn project(lb: []f64, ub: []f64, xk: []f64, gk: []f64, Bk: [][]f64, pk: []f64, tk: []f64, ts: []f64, xt: []f64, xc: []f64, Wk: []usize, ta: *usize) !void {
    const n: usize = xk.len;

    for (0..n) |i| pk[i] = -gk[i];
    checkbounds(xk, pk, tk, lb, ub);

    @memcpy(ts, tk); // use bf as buffer, for sorting
    insertionSort(ts); // sorted tk

    var tc: f64 = 0.0; // step size of the Cauchy point
    var dt: f64 = undefined;
    var dt_min: f64 = undefined;

    var df: f64 = undefined;
    var ddf: f64 = undefined;

    @memcpy(xt, xk); // x(t_0) and x( t(j-1) )

    for (ts) |trial| { // for j = 1, 2, ...
        if (tc == trial) continue;
        dt = trial - tc;

        @memcpy(xc, xt); // xc ← x( t(j-1) )

        for (0..n) |i| pk[i] = if (tc < tk[i]) -gk[i] else 0.0; // p(j-1)

        // pᵀ⋅( Bk⋅x( t(j-1) ) + gk ) = pᵀ⋅( (Lk⋅Rk)⋅x( t(j-1) ) + gk )
        try trmv('R', 'N', Bk, xc);
        try trmv('R', 'T', Bk, xc);
        for (0..n) |i| xc[i] += gk[i];
        df = try dot(pk, xc);

        // pᵀ⋅Bk⋅p = pᵀ⋅(Lk⋅Rk)⋅p
        try trmv('R', 'N', Bk, pk);
        ddf = try dot(pk, pk);
        dt_min = -df / ddf;

        if (dt <= dt_min) {
            for (0..n) |i| xt[i] -= if (trial <= tk[i]) dt * gk[i] else 0.0;
            tc = trial;
        } else break;
    }

    tc += @max(0.0, dt_min);
    ta.* = 0;

    for (0..n) |i| {
        if (tc < tk[i]) {
            xc[i] = xk[i] - tc * gk[i];
        } else {
            xc[i] = xk[i] - tk[i] * gk[i];
            Wk[ta.*] = i;
            ta.* += 1;
        }
    }
}

// Solve the KKT subspace quadratic problem for the line-search direction
//
// pk := buffer for KKT solved search direction
// xt := buffer for linear algebra
// xc := Cauchy point
// Ck := buffer for Ak⋅Hk⋅Akᵀ
fn solveKKT(xk: []f64, gk: []f64, Bk: [][]f64, pk: []f64, xt: []f64, xc: []f64, Ck: [][]f64, Wk: []usize, ta: usize) !void {
    if (ta == 0) return;
    const n: usize = xk.len;

    // Format Ak
    for (0..ta, Wk[0..ta]) |ia, ja| {
        @memset(Ck[ia], 0.0);
        Ck[ia][ja] = 1.0;
    }

    // Solve Hk⋅Akᵀ
    for (0..ta) |ia| try trsv('R', 'T', Bk, Ck[ia]);
    for (0..ta) |ia| try trsv('R', 'N', Bk, Ck[ia]);

    // Truncate to Ak⋅Hk⋅Akᵀ
    for (0..ta) |i| {
        for (0..ta, Wk[0..ta]) |j, ja| Ck[i][j] = Ck[i][ja];
    }

    try cholesky(Ck[0..ta]);

    // pk ← Bk⁻¹⋅∇f(xk)
    @memcpy(pk, gk);
    try trsv('R', 'T', Bk, pk);
    try trsv('R', 'N', Bk, pk);
    for (0..n) |i| pk[i] += xc[i] - xk[i]; // pk ← Bk⁻¹⋅∇f(xk) + (xc - xk)

    // Truncate to Ak⋅( Hk⋅gk + dk )
    for (0..ta, Wk[0..ta]) |i, ia| xt[i] = pk[ia]; // use xt as buffer

    // Solve λₖ
    try trsv('L', 'N', Ck[0..ta], xt[0..ta]); // L⋅y = b
    try trsv('L', 'T', Ck[0..ta], xt[0..ta]); // Lᵀ⋅x = y

    for (0..n) |i| pk[i] = -gk[i]; // pk ← -gk
    for (Wk[0..ta], xt[0..ta]) |ia, lambda_ia| pk[ia] += lambda_ia; // pk ← Akᵀ⋅λₖ - gk

    // Bk⁻¹⋅(Akᵀ⋅λₖ - gk)
    try trsv('R', 'T', Bk, pk);
    try trsv('R', 'N', Bk, pk);
}

test "Gradient Projection Method" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    var lb: [2]f64 = .{ -2.0, -1.0 };
    var ub: [2]f64 = .{ 0.8, 1.0 };
    var xk: [2]f64 = .{ -1.2, 1.0 };

    // Bk = Rᵀ⋅R = cholescky( exact Hessian at xk = .{ -1.2, 1.0 } )
    const Bk: [][]f64 = try ArrF64.matrix(2, 2);
    defer ArrF64.free(Bk);

    inline for (.{ 0x1.23c0d99c17436p+5, 0x1.a52d7f6fc5311p+3 }, Bk[0]) |v, *p| p.* = v;
    inline for (.{ 0x1.0000000000000p+0, 0x1.4b1d7f7c3508bp+2 }, Bk[1]) |v, *p| p.* = v;

    // gk = exact gradient at xk = .{ -1.2, 1.0 }
    var gk: [2]f64 = .{ -0x1.af33333333332p+7, -0x1.5ffffffffffffp+6 };

    // pk ← -∇f(xk)
    var pk: [2]f64 = undefined;
    for (&pk, gk) |*pk_i, gk_i| pk_i.* = -gk_i;

    // tk = breakpoints' step sizes
    var tk: [2]f64 = undefined;

    checkbounds(&lb, &ub, &xk, &pk, &tk);

    var ts: [2]f64 = undefined;
    var xt: [2]f64 = undefined;
    var xc: [2]f64 = undefined;
    var Wk: [2]usize = undefined;
    var ta: usize = undefined;

    try project(&lb, &ub, &xk, &gk, Bk, &pk, &tk, &ts, &xt, &xc, &Wk, &ta);

    // Ck = Ak⋅Hk⋅Akᵀ
    const Ck: [][]f64 = try ArrF64.matrix(2, 2);
    defer ArrF64.free(Ck);

    try solveKKT(&xk, &gk, Bk, &pk, &xt, &xc, Ck, &Wk, ta);

    for (&xt, xk, pk) |*xt_i, xk_i, pk_i| xt_i.* = xk_i + pk_i;
    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    try testing.expect(rosenbrock.func(&xt) <= rosenbrock.func(&xk));
}

fn insertionSort(a: []f64) void {
    if (a.len == 1) return;
    var j: usize = undefined;

    for (a[1..], 1..) |v, i| {
        j = i;
        while (0 < j and v < a[j - 1]) : (j -= 1) a[j] = a[j - 1];
        a[j] = v;
    }
}

fn dot(x: []f64, y: []f64) !f64 {
    if (x.len != y.len) return error.DimensionMismatch;
    var t: f64 = 0.0;
    for (x, y) |x_i, y_i| t += x_i * y_i;
    return t;
}

const INF: f64 = std.math.inf(f64);

const std = @import("std");
const testing = std.testing;

const trmv = @import("./linalg/trmv.zig").trmv;
const trsv = @import("./linalg/trsv.zig").trsv;
const cholesky = @import("./linalg/cholesky.zig").cholesky;

const Array = @import("./linalg/array.zig").Array;
const Rosenbrock = @import("./misc/Rosenbrock.zig");
