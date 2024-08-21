const Errors = error{DimensionMismatch};

pub fn trmv(comptime ul: u8, comptime tA: u8, A: [][]f64, x: []f64) !void {
    const n: usize = A.len;
    if (n != x.len) return error.DimensionMismatch;

    switch (ul) {
        'R', 'U' => {
            switch (tA) {
                'N' => {
                    // x ← R⋅x
                    var t: f64 = undefined;
                    for (0..n) |i| {
                        t = 0.0;
                        for (i..n) |j| t += A[i][j] * x[j];
                        x[i] = t;
                    }
                },
                'T' => {
                    // x ← Rᵀ⋅x
                    var t: f64 = undefined;
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        t = x[i];
                        x[i] = 0.0;
                        for (i..n) |j| x[j] += A[i][j] * t;
                        if (i == 0) break;
                    }
                },
                else => @compileError(""),
            }
        },
        'L' => @compileError("Not supported yet."),
        else => @compileError(""),
    }
}

test "trmv: x ← R⋅x, x ← Rᵀ⋅x" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const A: [][]f64 = try ArrF64.matrix(4, 4);
    defer ArrF64.free(A);

    const x: []f64 = try ArrF64.vector(4);
    defer ArrF64.free(x);

    inline for (.{ 2.0, 6.0, 8.0, 1.0 }, A[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 6.0, 1.0, 7.0, 5.0 }, A[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 8.0, 7.0, 4.0, 9.0 }, A[2]) |val, *ptr| ptr.* = val;
    inline for (.{ 1.0, 5.0, 9.0, 3.0 }, A[3]) |val, *ptr| ptr.* = val;

    const Z: [2][4]f64 = .{ .{ 7.1, 3.6, 3.5, 0.9 }, .{ 1.0, 3.7, 9.7, 6.7 } };

    inline for (.{ 'N', 'T' }, Z) |tA, z| {
        inline for (.{ 0.5, 0.7, 0.2, 0.3 }, x) |val, *ptr| ptr.* = val;
        try trmv('R', tA, A, x);
        for (x, &z) |x_i, z_i| try testing.expectApproxEqRel(x_i, z_i, 1e-15);
    }
}

const std = @import("std");
const testing = std.testing;
const Array = @import("./array.zig").Array;
