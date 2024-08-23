//! References
//! [1] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Algorithm 3.5, 3.6

const Errors = error{
    DescentDirectionError,
    DimensionMismatch,
    SearchError,
    ZoomError,
};

const WolfeParams = struct {
    LSC1: comptime_float = 1e-4, // line search factor 1
    LSC2: comptime_float = 9e-1, // line search factor 2
    SMAX: comptime_float = 65536.0, // line search max. step
    SMIN: comptime_float = 1e-2, // line search min. step
};

// fg := function and gradient
fn search(
    fg: anytype,
    xk: []f64,
    gk: []f64,
    pk: []f64,
    xn: []f64,
    gn: []f64,
    opt_smin: ?f64,
    opt_smax: ?f64,
    comptime params: WolfeParams,
) !void {
    const FG: type = comptime @TypeOf(fg);
    if (!@hasDecl(FG, "func") or !@hasDecl(FG, "grad")) @compileError("");

    const smin: f64 = if (opt_smin) |s| @max(params.SMIN, s) else params.SMIN;
    const smax: f64 = if (opt_smax) |s| @max(params.SMAX, s) else params.SMAX;

    const f0: f64 = fg.func(xk); // ϕ(0) = f(xk)
    const g0: f64 = try dot(pk, gk); // ϕ'(0) = pkᵀ⋅∇f(xk)

    if (0.0 <= g0) return error.DescentDirectionError;

    var f_old: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var a_old: f64 = 0.0;
    var a_now: f64 = smin;

    var iter: usize = 0;

    while (a_now <= smax) : (iter += 1) {
        for (xn, xk, pk) |*xn_i, xk_i, pm_i| xn_i.* = xk_i + a_now * pm_i; // xt ← xk + α⋅pk
        f_now = fg.func(xn); // ϕ(α) = f(xk + α⋅pk)

        // Test Wolfe conditions
        if ((f_now > f0 + params.LSC1 * a_now * g0) or (iter > 0 and f_now > f_old)) {
            return try zoom(fg, xk, pk, xn, gn, a_old, a_now, f0, g0, params);
        }

        fg.grad(xn, gn); // ∇f(xk + α⋅pk)
        g_now = try dot(pk, gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

        if (@abs(g_now) <= -params.LSC2 * g0) break; // return a_now
        if (0.0 <= g_now) return try zoom(fg, xk, pk, xn, gn, a_now, a_old, f0, g0, params);

        a_old = a_now;
        f_old = f_now;
        a_now = 2.0 * a_now;
    } else return error.SearchError;
}

fn zoom(
    fg: anytype,
    xk: []f64,
    pk: []f64,
    xn: []f64,
    gn: []f64,
    lb: f64,
    rb: f64,
    f0: f64,
    g0: f64,
    comptime params: WolfeParams,
) !void {
    const FG: type = comptime @TypeOf(fg);
    if (!@hasDecl(FG, "func") or !@hasDecl(FG, "grad")) @compileError("");

    var f_lo: f64 = undefined;
    var f_hi: f64 = undefined;

    var g_lo: f64 = undefined;
    var g_hi: f64 = undefined;

    var a_lo: f64 = lb;
    var a_hi: f64 = rb;

    var a_now: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var iter: usize = 0;

    for (xn, xk, pk) |*x_lo, xk_i, sm_i| x_lo.* = xk_i + a_lo * sm_i; // xt ← xk + α_lo⋅pk
    fg.grad(xn, gn); // ∇f(xk + α_lo⋅pk)

    f_lo = fg.func(xn); // ϕ(α_lo) = f(xk + α_lo⋅pk)
    g_lo = try dot(pk, gn); // ϕ'(α_lo) = pkᵀ⋅∇f(xk + α_lo⋅pk)

    for (xn, xk, pk) |*x_hi, xk_i, sm_i| x_hi.* = xk_i + a_hi * sm_i; // xt ← xk + α_hi⋅pk
    fg.grad(xn, gn); // ∇f(xk + α_hi⋅pk)

    f_hi = fg.func(xn); // ϕ(α_hi) = f(xk + α_hi⋅pk)
    g_hi = try dot(pk, gn); // ϕ'(α_hi) = pkᵀ⋅∇f(xk + α_hi⋅pk), ϕ'(α_hi) can be positive

    while (iter < 10) : (iter += 1) {
        // Interpolate α
        a_now = if (a_lo < a_hi)
            try interpolate(a_lo, a_hi, f_lo, f_hi, g_lo, g_hi)
        else
            try interpolate(a_hi, a_lo, f_hi, f_lo, g_hi, g_lo);

        for (xn, xk, pk) |*xn_i, xk_i, sm_i| xn_i.* = xk_i + a_now * sm_i; // xt ← xk + α⋅pk
        fg.grad(xn, gn); // ∇f(xk + α⋅pk)
        f_now = fg.func(xn); // ϕ(α) = f(xk + α⋅pk)
        g_now = try dot(pk, gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

        if ((f_now > f0 + params.LSC1 * a_now * g0) or (f_now > f_lo)) {
            a_hi = a_now;
            f_hi = f_now;
            g_hi = g_now;
        } else {
            if (@abs(g_now) <= -params.LSC2 * g0) break; // return a_now
            if (0.0 <= g_now * (a_hi - a_lo)) {
                a_hi = a_lo;
                f_hi = f_lo;
                g_hi = g_lo;
            }

            a_lo = a_now;
            f_lo = f_now;
            g_lo = g_now;
        }
    } else return error.ZoomError;
}

fn interpolate(a_old: f64, a_new: f64, f_old: f64, f_new: f64, g_old: f64, g_new: f64) !f64 {
    if (a_new <= a_old) return error.ZoomError;
    const d1: f64 = g_old + g_new - 3.0 * (f_old - f_new) / (a_old - a_new);
    const d2: f64 = @sqrt(d1 * d1 - g_old * g_new);
    const nu: f64 = g_new + d2 - d1;
    const de: f64 = g_new - g_old + 2.0 * d2;
    return a_new - (a_new - a_old) * (nu / de);
}

test "Gradient Projection Method" {
    var xk: [2]f64 = .{ -1.2, 1.0 };

    // gk = exact gradient at xk = .{ -1.2, 1.0 }
    var gk: [2]f64 = .{ -0x1.af33333333332p+7, -0x1.5ffffffffffffp+6 };

    // pk = KKT solution of gradient projection at xk
    var pk: [2]f64 = .{ 0x1.4bfdd81e2e596p-3, 0x0.0000000000000p+0 };

    var xn: [2]f64 = undefined;
    var gn: [2]f64 = undefined;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    try search(rosenbrock, &xk, &gk, &pk, &xn, &gn, 0.5, 1.0, .{});
    try testing.expect(rosenbrock.func(&xn) <= rosenbrock.func(&xk));
}

fn dot(x: []f64, y: []f64) !f64 {
    if (x.len != y.len) return error.DimensionMismatch;
    var t: f64 = 0.0;
    for (x, y) |x_i, y_i| t += x_i * y_i;
    return t;
}

const std = @import("std");
const testing = std.testing;

const Rosenbrock = @import("./misc/Rosenbrock.zig");
