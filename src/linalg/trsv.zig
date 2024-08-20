const Error = error{DimensionMismatch};

pub fn trsv(comptime ul: u8, comptime tA: u8, A: [][]f64, x: []f64) Error!void {
    const n: usize = A.len;
    if (n != x.len) return Error.DimensionMismatch;

    switch (ul) {
        'R', 'U' => {
            switch (tA) {
                'N' => {
                    // R⋅x = b
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        for (i + 1..n) |j| x[i] -= A[i][j] * x[j];
                        x[i] /= A[i][i];
                        if (i == 0) break;
                    }
                },
                'T' => {
                    // Rᵀ⋅x = b
                    for (0..n) |i| {
                        x[i] /= A[i][i];
                        for (i + 1..n) |j| x[j] -= A[i][j] * x[i];
                    }
                },
                else => @compileError(""),
            }
        },
        'L' => {
            switch (tA) {
                'N' => {
                    // L⋅x = b
                    var t: f64 = undefined;
                    for (0..n) |i| {
                        t = x[i];
                        for (0..i) |j| t -= A[i][j] * x[j];
                        x[i] = t / A[i][i];
                    }
                },
                'T' => {
                    // Lᵀ⋅x = b
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        x[i] /= A[i][i];
                        for (0..i) |j| x[j] -= A[i][j] * x[i];
                        if (i == 0) break;
                    }
                },
                else => @compileError(""),
            }
        },
        else => @compileError(""),
    }
}

test "trsv: x ← R⁻¹⋅x, x ← R⁻ᵀ⋅x, x ← L⁻¹⋅x, x ← L⁻ᵀ⋅x" {
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

    const Z: [2][4]f64 = .{
        .{ -0x1.affffffffffffp+1, 0x1.6ccccccccccccp+0, -0x1.6666666666666p-3, 0x1.9999999999999p-4 },
        .{ 0x1.0000000000000p-2, -0x1.999999999999ap-1, 0x1.e666666666668p-1, -0x1.8000000000001p+0 },
    };

    {
        const ul: u8 = 'R';
        inline for (.{ 'N', 'T' }, Z) |tA, z| {
            inline for (.{ 0.5, 0.7, 0.2, 0.3 }, x) |val, *ptr| ptr.* = val;
            try trsv(ul, tA, A, x);
            for (x, &z) |x_i, z_i| try testing.expectApproxEqRel(x_i, z_i, 1e-15);
        }
    }
    {
        const ul: u8 = 'L';
        inline for (.{ 'T', 'N' }, Z) |tA, z| {
            inline for (.{ 0.5, 0.7, 0.2, 0.3 }, x) |val, *ptr| ptr.* = val;
            try trsv(ul, tA, A, x);
            for (x, &z) |x_i, z_i| try testing.expectApproxEqRel(x_i, z_i, 1e-15);
        }
    }
}

const std = @import("std");
const testing = std.testing;
const Array = @import("./array.zig").Array;
