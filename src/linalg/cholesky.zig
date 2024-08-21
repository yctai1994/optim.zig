const Errors = error{NotPositiveDefinite};

pub fn cholesky(A: [][]f64) !void {
    const n: usize = A.len;
    var t: f64 = undefined;

    for (0..n) |i| {
        for (i..n) |j| {
            t = A[i][j];
            for (0..i) |k| t -= A[i][k] * A[j][k];
            if (i == j) {
                if (t <= 0.0) return Errors.NotPositiveDefinite;
                A[i][i] = @sqrt(t);
            } else A[j][i] = t / A[i][i];
        }
    }

    for (0..n) |j| {
        for (0..j) |i| A[i][j] = 0.0;
    }
}

test "Cholesky" {
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

const std = @import("std");
const testing = std.testing;
const Array = @import("./array.zig").Array;
