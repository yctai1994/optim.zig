//! References
//! [1] W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
//!     "Numerical Recipes 3rd Edition: The Art of Scientific Computing,"
//!     2007, Sec. 2.9, 2.10.1

const Errors = error{
    NotPositiveDefinite,
    DimensionMismatch,
    SingularError,
    NanError,
};

pub fn cholesky(A: [][]f64) !void {
    const n: usize = A.len;
    var t: f64 = undefined;

    for (0..n) |i| {
        for (i..n) |j| {
            t = A[i][j];
            for (0..i) |k| t -= A[i][k] * A[j][k];
            if (i == j) {
                if (t <= 0.0) return error.NotPositiveDefinite;
                A[i][i] = @sqrt(t);
            } else A[j][i] = t / A[i][i];
        }
    }

    for (0..n) |j| {
        for (0..j) |i| A[i][j] = 0.0;
    }
}

test "cholesky → L⋅Lᵀ" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const A: [][]f64 = try ArrF64.matrix(5, 5);
    defer ArrF64.free(A);

    inline for (.{ 5.0, -2.0, 0.0, -2.0, -2.0 }, A[0]) |val, *ptr| ptr.* = val;
    inline for (.{ -2.0, 5.0, -2.0, 0.0, 0.0 }, A[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, -2.0, 5.0, -2.0, 0.0 }, A[2]) |val, *ptr| ptr.* = val;
    inline for (.{ -2.0, 0.0, -2.0, 5.0, -2.0 }, A[3]) |val, *ptr| ptr.* = val;
    inline for (.{ -2.0, 0.0, 0.0, -2.0, 5.0 }, A[4]) |val, *ptr| ptr.* = val;

    const Z: struct { [1]f64, [2]f64, [3]f64, [4]f64, [5]f64 } = .{
        .{0x1.1e3779b97f4a8p+1},
        .{ -0x1.c9f25c5bfedd9p-1, 0x1.06526aa25a13bp+1 },
        .{ 0x0.0000000000000p0, -0x1.f3a92ca2f4b7bp-1, 0x1.0184f0ebef245p+1 },
        .{ -0x1.c9f25c5bfedd9p-1, -0x1.8fba8a1bf6f95p-2, -0x1.2ef6c11591d06p+0, 0x1.a081a89bc7e07p+0 },
        .{ -0x1.c9f25c5bfedd9p-1, -0x1.8fba8a1bf6f95p-2, -0x1.83cb34967d339p-3, -0x1.f3ceca548973bp+0, 0x1.c9f25c5bfedddp-2 },
    };

    try cholesky(A);
    inline for (Z, 0..) |z, i| {
        for (z, 0..) |z_j, j| try testing.expectApproxEqRel(z_j, A[i][j], 1e-15);
    }
}

fn update(R: [][]f64, u: []f64, v: []f64, b: []f64) !void {
    const n: usize = u.len;
    if (n != v.len) return error.DimensionMismatch;

    // Find largest k such that u[k] ≠ 0.
    var k: usize = 0;
    for (b, u, 0..) |*b_i, u_i, i| {
        if (u_i != 0.0) k = i;
        b_i.* = u_i;
    }

    // Transform R + u⋅vᵀ to upper Hessenberg.
    if (0 < k) {
        var i: usize = k - 1;
        while (0 <= i) : (i -= 1) {
            rotate(R, i, n, b[i], -b[i + 1]);
            b[i] = try apy2(b[i], b[i + 1]);
            if (i == 0) break;
        }
    }

    for (R[0], v) |*R_0i, v_i| R_0i.* += b[0] * v_i;

    // Transform upper Hessenberg matrix to upper triangular.
    for (0..k, 1..) |j, jp1| rotate(R, j, n, R[j][j], -R[jp1][j]);

    // Check singularity.
    for (R, 0..) |R_j, j| if (R_j[j] == 0.0) return error.SingularError;
}

fn rotate(R: [][]f64, i: usize, n: usize, a: f64, b: f64) void {
    var c: f64 = undefined;
    var s: f64 = undefined;
    var w: f64 = undefined;
    var y: f64 = undefined;
    var f: f64 = undefined;

    if (a == 0.0) {
        c = 0.0;
        s = if (b < 0.0) -1.0 else 1.0;
    } else if (@abs(a) > @abs(b)) {
        f = b / a;
        c = copysign(1.0 / @sqrt(1.0 + pow2(f)), a);
        s = f * c;
    } else {
        f = a / b;
        s = copysign(1.0 / @sqrt(1.0 + pow2(f)), b);
        c = f * s;
    }

    for (R[i][i..n], R[i + 1][i..n]) |*R_ij, *R_ip1j| {
        y = R_ij.*;
        w = R_ip1j.*;
        R_ij.* = c * y - s * w;
        R_ip1j.* = s * y + c * w;
    }
}

test "update Rᵀ⋅R" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const R: [][]f64 = try ArrF64.matrix(3, 3);
    defer ArrF64.free(R);

    inline for (.{ 2.0, 6.0, 8.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, R[2]) |val, *ptr| ptr.* = val;

    const u: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(u);

    const v: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(v);

    const b: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(b);

    inline for (.{ 1.0, 5.0, 3.0 }, u) |val, *ptr| ptr.* = val;
    inline for (.{ 2.0, 3.0, 1.0 }, v) |val, *ptr| ptr.* = val;

    try update(R, u, v, b);

    const A: [3][3]f64 = .{ // answers
        .{ 0x1.8a85c24f70658p+03, 0x1.44715e1c46896p+04, 0x1.be6ef01685ec3p+03 },
        .{ -0x1.0000000000000p-52, 0x1.4e2ba31c14a89p+01, 0x1.28c0f1b618468p+02 },
        .{ 0x0.0000000000000p+00, 0x0.0000000000000p+00, -0x1.dd36445718509p-01 },
    };

    for (A, R) |A_i, R_i| try testing.expect(std.mem.eql(f64, &A_i, R_i));
}

fn apy2(x: f64, y: f64) !f64 {
    // nan case
    if (x != x or y != y) return error.NanError;

    // general case
    const X: f64 = @abs(x);
    const Y: f64 = @abs(y);
    const w: f64 = @max(X, Y);
    const z: f64 = @min(X, Y);

    return if (z == 0.0) w else w * @sqrt(1.0 + pow2(z / w));
}

inline fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const testing = std.testing;
const copysign = std.math.copysign;

const Array = @import("./array.zig").Array;
